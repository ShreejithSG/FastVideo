# GLM-img Porting Guide: Achieving SSIM > 0.9

## Problem Statement

After porting GLM-img from sglang (PR #16894) to FastVideo and swapping some components with FastVideo's implementations, the SSIM dropped to **0.4** (critically low). This guide helps identify and fix the numerical alignment issues.

## Understanding GLM-img Architecture

### Key Components (from sglang implementation)

1. **DiT Transformer** (`glm_image.py`)
   - 30 layers
   - 32 attention heads with 128 dim per head
   - Uses KV caching for efficiency
   - RoPE (Rotary Position Embedding)
   - Prior token conditioning
   
2. **VAE** (AutoencoderKL)
   - 16 input/output channels
   - Scale factor: 8
   - Latent statistics: mean/std normalization
   
3. **Text Encoder**
   - 1472-dim text embeddings
   - Condition dim: 256

4. **Scheduler**
   - Flow matching-based
   - Timestep embedding: 512 dim

## Critical Implementation Details to Preserve

### 1. **Exact Numerical Precision**

```python
# sglang uses specific dtypes
vae_precision: str = "bf16"
enable_autocast: bool = False  # Important!
```

**Action**: Do NOT enable autocast if sglang doesn't use it!

### 2. **KV Cache Implementation**

GLM-img uses a custom KV cache system with three modes:
- `"write"`: Cache conditioning tokens  
- `"read"`: Use cached conditioning
- `"skip"`: Skip caching for negative prompt

```python
class GlmImageKVCache:
    def set_mode(self, mode: Optional[str]):
        # modes: "write", "read", "skip"
```

**Action**: If swapping with FastVideo's attention, ensure KV cache semantics match exactly!

### 3. **Rotary Embedding (RoPE)**

```python
# sglang applies RoPE to Q and K
def _apply_rotary_emb(q, k, freqs_cis):
    # Must match exactly!
```

**Action**: Verify FastVideo's RoPE implementation matches sglang's frequency calculation.

### 4. **VAE Latent Normalization**

```python
# sglang uses specific latent statistics
latents_mean = tensor(config.latents_mean)  
latents_std = tensor(config.latents_std)
latents = (latents - latents_mean) / latents_std
```

**Action**: Ensure these statistics are applied in the same order and precision!

### 5. **Prior Token Handling**

GLM-img uses a VQ codebook for prior conditioning:
```python
prior_vq_quantizer_codebook_size: int = 16384
prior_token_id: tensor  # from batch
prior_token_drop: float  # dropout rate
```

**Action**: This is unique to GLM-img - must be preserved exactly!

## Common SSIM Degradation Causes

### Cause 1: Component Swapping Issues (70% likely)

**Problem**: You replaced sglang's attention with FastVideo's attention backend.

**Symptoms**:
- SSIM < 0.5
- Different attention backends (FLASH_ATTN vs TORCH_SDPA)
- Missing KV cache modes

**Fix**:
```python
# Test attention in isolation
from sglang.multimodal_gen.runtime.layers.attention import USPAttention as SGL_Attn
from fastvideo.attention import FastVideoAttention as FV_Attn

# Create identical inputs
q, k, v = create_test_tensors()

# Compare outputs
sgl_output = SGL_Attn(...)(q, k, v)
fv_output = FV_Attn(...)(q, k, v)

assert_close(sgl_output, fv_output, atol=1e-2, rtol=1e-2)
```

### Cause 2: Scheduler/Timestep Mismatch (20% likely)

**Problem**: Different scheduler implementation or timestep computation.

**Symptoms**:
- SSIM in 0.4-0.7 range
- Noise scheduling differences

**Fix**:
```python
# Verify timesteps match exactly
sgl_scheduler.set_timesteps(num_steps)
fv_scheduler.set_timesteps(num_steps)

assert torch.equal(sgl_scheduler.timesteps, fv_scheduler.timesteps)
```

### Cause 3: VAE Encode/Decode Precision (10% likely)

**Problem**: VAE using different precision or normalization.

**Symptoms**:
- SSIM in 0.6-0.8 range
- Visual artifacts in generated images

**Fix**:
```python
# Test VAE in isolation
image = load_reference_image()
sgl_latent = sgl_vae.encode(image)
fv_latent = fv_vae.encode(image)

assert_close(sgl_latent, fv_latent, atol=1e-3, rtol=1e-3)
```

## Step-by-Step Debugging Process

### Step 1: Baseline Test

Run the diagnostic tool:
```bash
cd fastvideo/tests/alignment
python diagnose_ssim.py --ssim 0.4 --model glm-img --precision bf16
```

### Step 2: Component Isolation

Test each component independently:

```python
# Test 1: VAE
pytest fastvideo/tests/alignment/test_glm_img_vae.py -v

# Test 2: Text Encoder  
pytest fastvideo/tests/alignment/test_glm_img_encoder.py -v

# Test 3: DiT Transformer
pytest fastvideo/tests/alignment/test_glm_img_dit.py -v

# Test 4: Scheduler
pytest fastvideo/tests/alignment/test_scheduler_alignment.py -v
```

### Step 3: Identify the Failing Component

The component with the highest error is likely the root cause.

### Step 4: Deep Dive into Failing Component

If DiT fails (most likely):

```python
# Compare layer by layer
for layer_idx in range(num_layers):
    sgl_layer_output = sgl_dit.layers[layer_idx](...)
    fv_layer_output = fv_dit.layers[layer_idx](...)
    
    diff = torch.abs(sgl_layer_output - fv_layer_output).max()
    if diff > threshold:
        print(f"Layer {layer_idx} mismatch: {diff}")
        # Investigate this layer's attention, feedforward, etc.
```

### Step 5: Fix and Verify

After fixing, verify SSIM improves:

```python
# Run E2E test
pytest fastvideo/tests/alignment/test_pipeline_alignment.py \
    --model glm-img --precision bf16 --num-steps 20
```

## Expected SSIM After Fixes

| Fix Applied | Expected SSIM | Notes |
|-------------|---------------|-------|
| No fixes | 0.4 | Current state |
| VAE fix only | 0.5-0.6 | Minor improvement |
| Attention fix | 0.7-0.85 | Major improvement |
| Scheduler fix | 0.8-0.9 | Significant improvement |
| All components exact | **0.93-0.98** | Target for bf16 |
| All + fp32 | **0.98-1.0** | Ideal state |

## Specific Recommendations for Your Case

Based on SSIM=0.4 (critical), most likely issues in order:

1. **Attention backend mismatch** (80% confidence)
   - Check: Did you swap USPAttention with FastVideo's attention?
   - Fix: Either use sglang's attention OR carefully port KV cache logic

2. **Missing KV cache modes** (60% confidence)
   - Check: Does your attention support "write"/"read"/"skip" modes?
   - Fix: Implement GlmImageKVCache exactly as in sglang

3. **RoPE frequency mismatch** (40% confidence)
   - Check: Compare freqs_cis computation
   - Fix: Use sglang's exact RoPE implementation

4. **Prior token handling** (30% confidence)
   - Check: Is prior_token_id being passed and embedded correctly?
   - Fix: Port GlmImage's prior token embedding exactly

## Testing Strategy

### Phase 1: Component Tests (Days 1-2)
- [ ] Create test_glm_img_vae.py
- [ ] Create test_glm_img_encoder.py  
- [ ] Create test_glm_img_dit.py
- [ ] Run all component tests

### Phase 2: Integration Tests (Days 3-4)
- [ ] Test DiT with real text embeddings
- [ ] Test full denoising loop
- [ ] Test KV caching behavior

### Phase 3: E2E Validation (Day 5)
- [ ] Generate test images
- [ ] Compute SSIM against sglang reference
- [ ] Target: SSIM > 0.93

## Code Templates

### Template: GLM-img DiT Alignment Test

```python
# fastvideo/tests/alignment/test_glm_img_dit.py
import pytest
import torch
from torch.testing import assert_close

# Import sglang reference
from sglang.multimodal_gen.runtime.models.dits.glm_image import GlmImageTransformer as SGL_DiT

# Import FastVideo port
from fastvideo.models.dits.glm_image import GlmImageTransformer as FV_DiT

def test_glm_img_dit_forward():
    """Test DiT forward pass alignment."""
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    # Load both models
    sgl_dit = SGL_DiT.from_pretrained(...).to(device, dtype).eval()
    fv_dit = FV_DiT.from_pretrained(...).to(device, dtype).eval()
    
    # Create test inputs
    hidden_states = torch.randn(1, 16, 64, 64, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1, 77, 1472, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device)
    
    # Forward pass
    with torch.no_grad():
        sgl_output = sgl_dit(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            kv_caches_mode="write",
        )
        
        fv_output = fv_dit(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            kv_caches_mode="write",
        )
    
    # Compare
    assert_close(sgl_output, fv_output, atol=1e-2, rtol=1e-2)
    print("✓ GLM-img DiT alignment test passed!")
```

## Success Criteria

✅ **Success**: SSIM > 0.93 for bf16, > 0.95 for fp32
⚠️ **Acceptable**: SSIM > 0.90 for bf16  
❌ **Failure**: SSIM < 0.90 - indicates remaining alignment issues

## Questions to Ask

When debugging SSIM=0.4:

1. **Did I use the exact same model weights?**
   - Check: `model.load_state_dict(torch.load(...))`
   - Verify: Weight checksums match

2. **Did I use the exact same random seed?**
   - Check: `torch.manual_seed(42)` called before generation
   - Verify: Same seed produces same noise

3. **Did I swap any components?**
   - Check: List all replaced modules
   - Verify: Each replacement is numerically equivalent

4. **Are precision settings identical?**
   - Check: bf16 vs fp32 vs fp16
   - Verify: No unexpected autocasting

5. **Are all hyperparameters identical?**
   - Check: num_steps, guidance_scale, etc.
   - Verify: Scheduler config matches

## Next Actions

1. **Immediate**: Run `diagnose_ssim.py` to get baseline metrics
2. **Day 1**: Create and run component-level tests
3. **Day 2**: Identify and fix failing component
4. **Day 3**: Re-run E2E and verify SSIM > 0.9
5. **Day 4**: Document fixes and add regression tests

## Contact for Help

If SSIM remains < 0.9 after following this guide:
- Check sglang implementation details again
- Compare intermediate outputs layer-by-layer
- Consider using sglang's exact modules without swapping
- File an issue with detailed alignment metrics

---

**Remember**: SSIM of 1.0 is only possible with **identical** implementations (same weights, same operations, same precision, same random state). SSIM of 0.93-0.98 is realistic for bf16 with correct implementation!
