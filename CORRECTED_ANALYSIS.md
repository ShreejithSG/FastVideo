# GLM-Image SSIM Investigation - Corrected Analysis

## My Previous Mistake - Apology

I incorrectly assumed dim=1 was correct for KV cache. You were absolutely right:
- **dim=2 is CORRECT** (improved SSIM from 0.4 to 0.66)
- **dim=1 made it WORSE** (back to 0.441)
- I misunderstood the tensor layout

## Understanding the Tensor Shapes

Let me reconsider the actual tensor flow after unflatten:

### Option A: If using view/reshape directly
```python
# After projection: [B, S, embed_dim]
key = self.to_k(hidden_states)  # [B, S, num_heads * head_dim]

# After unflatten(2, (heads, head_dim)):
key = key.unflatten(2, (self.heads, -1))  # [B, S, heads, head_dim]

# This gives: dim=0=batch, dim=1=sequence, dim=2=heads, dim=3=head_dim
```

### Option B: If attention expects [B, H, S, D]
```python
# If there's a permute after unflatten:
key = key.permute(0, 2, 1, 3)  # [B, heads, S, head_dim]

# Then: dim=0=batch, dim=1=heads, dim=2=sequence, dim=3=head_dim
# In this case, dim=2 IS the sequence dimension!
```

**Since dim=2 works, Option B must be correct!** There must be a permute happening.

## Actual Issues to Fix (Per Your Request)

### 1. Attention Mask Handling (Priority 1)

**Current Issue in bc570af**:
```python
# Lines ~278-285 in glm_image_before_denoising.py
if L_pos < max_L:
    att_mask_pos[:, L_pos:] = 0  # Padding mask

# But is this mask actually used by the attention layer?
```

**What to Check**:
1. Does `GlmImageAttention` actually receive and use `attention_mask`?
2. Is the mask shape correct for LocalAttention?
3. Are padding tokens properly masked out?

**Potential Fix**:
```python
# In GlmImageAttention.forward()
def forward(self, ..., attention_mask=None):
    # ...
    if attention_mask is not None:
        # Expand mask to [B, H, S_q, S_k] if needed
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Apply mask before attention
        hidden_states = self.attn(query, key, value, attention_mask=attention_mask)
```

### 2. VAE Latent Normalization (Priority 1)

**Current Question**: Are VAE latents normalized correctly?

Check these files:
- `fastvideo/configs/models/vaes/glm_image.py` - latent mean/std config
- `fastvideo/pipelines/stages/glm_image_decoding.py` - VAE decode
- Any pre/post processing of latents

**What to Verify**:
```python
# Encoding (if used):
latents = (latents - latents_mean) / latents_std

# Decoding:
latents = latents * latents_std + latents_mean
```

**Compare with sglang**:
- Check if sglang uses specific latent statistics
- Verify order of operations (normalize before or after other transforms)

### 3. Glyph Embedding Details (Priority 2)

**bc570af improved glyph extraction**, but check:
- Are glyph embeddings positioned correctly?
- Is padding handled the same way as sglang?
- Does the flattening + masking match sglang exactly?

```python
# Current in bc570af:
prompt_embeds = prompt_outputs.last_hidden_state[valid_mask].unsqueeze(0).to(dtype)

# Verify this matches sglang's approach
```

## Investigation Checklist

### Step 1: Understand Tensor Layout
- [ ] Check if there's a permute after unflatten in attention
- [ ] Confirm dim=2 is indeed sequence dimension after permute
- [ ] Document the actual tensor flow

### Step 2: Attention Mask Analysis
- [ ] Print attention_mask shape in forward pass
- [ ] Verify mask is passed to LocalAttention
- [ ] Check if mask format matches expected input
- [ ] Compare mask construction with sglang

### Step 3: VAE Pipeline Check
- [ ] Review VAE config for latent statistics
- [ ] Check encoding/decoding normalization
- [ ] Compare with sglang's VAE handling
- [ ] Verify no missing preprocessing steps

### Step 4: Test and Iterate
- [ ] Make targeted fix to attention mask
- [ ] Test SSIM (expect 0.66 → 0.75-0.85)
- [ ] If needed, fix VAE normalization
- [ ] Test SSIM (expect 0.85-0.92)
- [ ] Fine-tune remaining issues

## Code Locations to Check

### Attention Implementation
```
fastvideo/models/dits/glm_image.py
- Line ~304: GlmImageAttention class
- Line ~360: forward() method
- Line ~378: unflatten operation
- Check if there's a permute after this
```

### Before Denoising (Text Processing)
```
fastvideo/pipelines/stages/glm_image_before_denoising.py
- Line ~183: glyph text extraction
- Line ~244: prompt embedding flattening
- Line ~278: attention mask construction
- Line ~290: batch.attention_mask assignment
```

### VAE Pipeline
```
fastvideo/configs/models/vaes/glm_image.py
- Check for latents_mean, latents_std

fastvideo/pipelines/stages/glm_image_decoding.py
- Check decode preprocessing
```

## Expected SSIM Path

```
0.4 (initial broken)
  ↓
0.66 (bc570af: fixed glyphs + KV cache dim=2) ← Current
  ↓
0.75-0.85 (fix attention mask)
  ↓
0.85-0.92 (fix VAE normalization)
  ↓
0.9+ (fine-tuning)
```

## Next Steps

1. **Don't revert dim=2!** It was correct.
2. Focus on attention mask handling first
3. Then check VAE latent normalization
4. Test incrementally

## Questions to Answer

1. **Tensor Layout**: After unflatten, is there a permute to [B, H, S, D]?
2. **Attention Mask**: Does LocalAttention receive and use the mask?
3. **VAE Stats**: What are latents_mean and latents_std values?
4. **Mask Shape**: Does mask match expected shape for attention?

## How I Can Help

Since I can't access your actual feature branch easily from here, I recommend:
1. Share the specific code sections (attention forward, mask handling)
2. Let me know what you find about tensor shapes
3. I'll help analyze and suggest targeted fixes

Again, my apologies for the confusion about dim=1 vs dim=2!
