# GLM-Image SSIM Fixes: Attention Mask and VAE Normalization

## Implementation Guide Based on sglang Analysis

### Fix 1: Attention Mask (HIGH PRIORITY - Expected +0.10-0.15 SSIM)

#### What sglang Does

```python
# In GlmImageAttention.forward() - sglang implementation
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,  # <-- ADD THIS
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[GlmImageLayerKVCache] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ... QKV projections, RoPE, KV cache ...
    
    # ATTENTION MASK CONSTRUCTION (ADD THIS BLOCK)
    if attention_mask is not None:
        text_attn_mask = attention_mask
        assert text_attn_mask.dim() == 2, \
            "the shape of text_attn_mask should be (batch_size, text_seq_length)"
        text_attn_mask = text_attn_mask.float().to(query.device)
        
        # Create full mask for text + image tokens
        mix_attn_mask = torch.ones(
            (batch_size, text_seq_length + image_seq_length), 
            device=query.device
        )
        mix_attn_mask[:, :text_seq_length] = text_attn_mask
        
        # Create 2D attention mask matrix
        mix_attn_mask = mix_attn_mask.unsqueeze(2)  # [B, total_seq, 1]
        attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)  # [B, total_seq, total_seq]
        attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)  # [B, 1, total_seq, total_seq]
    
    # Pass mask to attention
    hidden_states = self.attn(query, key, value, attention_mask=attention_mask)
```

#### How to Apply to FastVideo

**File**: `fastvideo/models/dits/glm_image.py`
**Location**: `GlmImageAttention.forward()` around line 360-420

**Step 1**: Add attention_mask parameter to forward signature
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[GlmImageLayerKVCache] = None,
    attention_mask: Optional[torch.Tensor] = None,  # ADD THIS LINE
) -> Tuple[torch.Tensor, torch.Tensor]:
```

**Step 2**: Add mask construction after KV cache handling (around line 408)
```python
# 4. KV Cache handling
# ... existing KV cache code ...

# 5. Attention mask construction (ADD THIS BLOCK)
if attention_mask is not None:
    text_attn_mask = attention_mask
    # Ensure mask is 2D: [batch_size, text_seq_length]
    assert text_attn_mask.dim() == 2, \
        f"Expected 2D attention mask, got {text_attn_mask.dim()}D"
    text_attn_mask = text_attn_mask.float().to(query.device)
    
    # Create combined mask for text + image tokens
    batch_size = text_attn_mask.size(0)
    total_seq_length = text_seq_length + image_seq_length
    mix_attn_mask = torch.ones(
        (batch_size, total_seq_length), 
        device=query.device,
        dtype=text_attn_mask.dtype
    )
    # Copy text mask to beginning
    mix_attn_mask[:, :text_seq_length] = text_attn_mask
    
    # Create 2D attention mask matrix
    mix_attn_mask = mix_attn_mask.unsqueeze(2)  # [B, total_seq, 1]
    attn_mask_matrix = mix_attn_mask @ mix_attn_mask.transpose(1, 2)  # [B, total_seq, total_seq]
    attention_mask = (attn_mask_matrix > 0).unsqueeze(1).to(query.dtype)  # [B, 1, total_seq, total_seq]

# 6. Attention computation with mask
hidden_states = self.attn(query, key, value, attention_mask=attention_mask)
```

**Step 3**: Propagate mask through transformer block

**File**: `fastvideo/models/dits/glm_image.py`
**Location**: `GlmImageTransformerBlock.forward()` around line 494-505

```python
attn_hidden_states, attn_encoder_hidden_states = self.attn1(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=norm_encoder_hidden_states,
    image_rotary_emb=image_rotary_emb,
    kv_cache=kv_cache,
    attention_mask=attention_kwargs.get('attention_mask') if attention_kwargs else None,  # ADD THIS
    **attention_kwargs,
)
```

### Fix 2: VAE Normalization (HIGH PRIORITY - Expected +0.10-0.15 SSIM)

#### What sglang Does

```python
# In pipeline config:
def get_decode_scale_and_shift(self, device, dtype, vae):
    latents_mean = (
        torch.tensor(self.vae_config.latents_mean)
        .view(1, self.vae_config.latent_channels, 1, 1)
        .to(device, dtype)
    )
    latents_std = (
        torch.tensor(self.vae_config.latents_std)
        .view(1, self.vae_config.latent_channels, 1, 1)
        .to(device, dtype)
    )
    return 1.0 / latents_std, latents_mean

# Before VAE decode:
# latents = latents * scale + shift
# Which is equivalent to: latents = latents * (1.0 / latents_std) + latents_mean
```

#### How to Apply to FastVideo

**Step 1**: Add latents_mean and latents_std to VAE config

**File**: `fastvideo/configs/models/vaes/glm_image.py`

```python
@dataclass
class GlmImageVAEArchConfig(VAEArchConfig):
    """Architecture config for GLM-Image VAE (AutoencoderKL)."""
    
    # ... existing fields ...
    
    # Add these fields:
    latents_mean: tuple[float, ...] | None = None
    latents_std: tuple[float, ...] | None = None
```

**Step 2**: Load values from model config

These values should come from the model's config.json on HuggingFace. You'll need to check the actual model repo for these values. Typical values might be:
- If not specified in config, they might be None (no normalization needed)
- If specified, they're typically 16-element tuples (one per latent channel)

**Step 3**: Apply normalization in decoding stage

**File**: `fastvideo/pipelines/stages/glm_image_decoding.py`

Before the VAE decode call, add:
```python
def forward(self, batch: GlmImageBatch) -> GlmImageBatch:
    latents = batch.denoised_latents
    
    # Apply VAE latent normalization if configured
    if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype
        ).view(1, -1, 1, 1)
        
        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype
        ).view(1, -1, 1, 1)
        
        # Denormalize: latents = latents * std + mean
        latents = latents * latents_std + latents_mean
    
    # Now decode
    images = self.vae.decode(latents)
    # ...
```

### Testing Strategy

#### Test After Each Fix

```bash
# After attention mask fix:
pytest fastvideo/tests/ssim/test_glm_image_similarity.py -v
# Expected: SSIM ~0.75-0.81 (up from 0.66)

# After VAE normalization fix:
pytest fastvideo/tests/ssim/test_glm_image_similarity.py -v
# Expected: SSIM ~0.85-0.92
```

#### Debug Prints to Verify

```python
# In attention forward:
if attention_mask is not None:
    print(f"Attention mask applied: shape={attention_mask.shape}, "
          f"zeros={(attention_mask==0).sum()}, ones={(attention_mask>0).sum()}")

# In decoding:
if latents_mean is not None:
    print(f"VAE normalization applied: mean_val={latents_mean.mean().item():.4f}, "
          f"std_val={latents_std.mean().item():.4f}")
    print(f"Latents before norm: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
    latents = latents * latents_std + latents_mean
    print(f"Latents after norm: min={latents.min().item():.4f}, max={latents.max().item():.4f}")
```

### Expected Results

| Fix | Code Location | Expected SSIM | Cumulative |
|-----|--------------|---------------|------------|
| Baseline | bc570af | 0.66 | 0.66 |
| + Attention mask | glm_image.py:360-420 | +0.10-0.15 | 0.76-0.81 |
| + VAE normalization | glm_image_decoding.py | +0.10-0.15 | 0.86-0.96 |

### Common Issues and Solutions

#### Issue 1: LocalAttention doesn't accept attention_mask

**Symptom**: TypeError about unexpected keyword argument

**Solution**: Check LocalAttention signature in `fastvideo/attention/layer.py`. You may need to:
1. Pass mask differently based on backend
2. Or modify LocalAttention to accept mask parameter

#### Issue 2: latents_mean/std values not available

**Symptom**: Config fields are None

**Solution**: 
1. Check model's config.json on HuggingFace for these values
2. If not present, they might not be needed (standard scaling_factor is sufficient)
3. Or compute them from reference implementation

#### Issue 3: Mask shape mismatch

**Symptom**: RuntimeError about tensor shapes

**Solution**: Verify shapes at each step:
```python
print(f"text_attn_mask: {text_attn_mask.shape}")  # Should be [B, text_seq]
print(f"mix_attn_mask: {mix_attn_mask.shape}")     # Should be [B, total_seq]
print(f"attn_mask_matrix: {attn_mask_matrix.shape}") # Should be [B, total_seq, total_seq]
print(f"final attention_mask: {attention_mask.shape}") # Should be [B, 1, total_seq, total_seq]
```

### Summary

**Critical Changes**:
1. Add `attention_mask` parameter to `GlmImageAttention.forward()`
2. Implement mask matrix construction matching sglang
3. Add `latents_mean` and `latents_std` to VAE config
4. Apply VAE normalization before decode

**Priority**: Both fixes are HIGH priority
**Estimated time**: 1-2 hours implementation + testing
**Expected outcome**: SSIM 0.66 → 0.85-0.92
