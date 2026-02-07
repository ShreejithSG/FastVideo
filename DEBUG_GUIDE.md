# Debugging Guide: Attention Mask and VAE Normalization

## Quick Diagnostic Commands

### 1. Check Tensor Shapes in Attention
Add these debug prints to `fastvideo/models/dits/glm_image.py`:

```python
# In GlmImageAttention.forward(), around line 377
def forward(self, hidden_states, encoder_hidden_states, ...):
    # After unflatten (line ~380)
    query = query.unflatten(2, (self.heads, -1))
    key = key.unflatten(2, (self.heads, -1))
    value = value.unflatten(2, (self.heads, -1))
    
    # ADD THIS DEBUG:
    print(f"[DEBUG] After unflatten:")
    print(f"  query shape: {query.shape}")  # Should be [B, S, H, D] or [B, H, S, D]?
    
    # Check if there's a permute here
    # Look for: query = query.permute(...) or similar
```

### 2. Check Attention Mask Usage
```python
# In GlmImageAttention.forward(), around line 411
hidden_states = self.attn(query, key, value)

# ADD THIS:
print(f"[DEBUG] Attention call:")
print(f"  Has attention_mask param? {attention_mask is not None if 'attention_mask' in locals() else 'Not defined'}")
```

### 3. Check VAE Latent Statistics
```python
# In decoding stage or VAE forward
print(f"[DEBUG] Latents before decode:")
print(f"  Mean: {latents.mean()}, Std: {latents.std()}")
print(f"  Min: {latents.min()}, Max: {latents.max()}")
```

## Issue 1: Attention Mask Not Being Used

### Symptom
Attention mask is constructed but not passed to attention layer.

### How to Check
```python
# In glm_image_before_denoising.py, end of forward():
print(f"[DEBUG] Batch attention_mask: {batch.attention_mask.shape if batch.attention_mask is not None else None}")

# In glm_image.py GlmImageTransformer2DModel.forward():
print(f"[DEBUG] Received attention_mask: {attention_mask.shape if attention_mask is not None else None}")

# In GlmImageTransformerBlock.forward():
print(f"[DEBUG] Block attention_mask: {attention_kwargs.get('attention_mask') if attention_kwargs else None}")
```

### Potential Fix
```python
# In GlmImageAttention.forward():
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[GlmImageLayerKVCache] = None,
    attention_mask: Optional[torch.Tensor] = None,  # ADD THIS
) -> Tuple[torch.Tensor, torch.Tensor]:
    # ...
    
    # Before calling self.attn, pass mask:
    hidden_states = self.attn(query, key, value, attention_mask=attention_mask)
```

## Issue 2: Attention Mask Wrong Shape

### Symptom
LocalAttention expects mask in specific shape but receives different.

### How to Check
```python
# Check LocalAttention signature in fastvideo/attention/layer.py
# Look for the expected mask shape in forward()

# In GlmImageAttention, before calling attn:
if attention_mask is not None:
    print(f"[DEBUG] Mask shape before attn: {attention_mask.shape}")
    print(f"[DEBUG] Query shape: {query.shape}")
    print(f"[DEBUG] Key shape: {key.shape}")
```

### Potential Fix
```python
# Reshape mask to match attention backend expectations
if attention_mask is not None:
    # LocalAttention might expect [B, S] or [B, 1, S_q, S_k]
    # Check and reshape accordingly
    if attention_mask.dim() == 2:
        # Expand to [B, 1, 1, S] for broadcasting
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
```

## Issue 3: VAE Latent Normalization Missing

### Symptom
Latents have unexpected value ranges or statistics.

### How to Check
```python
# Compare latent statistics with sglang expectations
# Check config file for expected values

# In decoding stage:
print(f"[DEBUG] Latents before VAE decode:")
print(f"  Shape: {latents.shape}")
print(f"  Mean: {latents.mean():.4f}, Std: {latents.std():.4f}")
print(f"  Range: [{latents.min():.4f}, {latents.max():.4f}]")

# After decode:
print(f"[DEBUG] Images after VAE decode:")
print(f"  Mean: {images.mean():.4f}, Std: {images.std():.4f}")
```

### Check Config
```python
# In fastvideo/configs/models/vaes/glm_image.py
# Look for:
# latents_mean = [...]
# latents_std = [...]

# Or in pipeline config:
# vae_config.latents_mean
# vae_config.latents_std
```

### Potential Fix
```python
# If latent normalization is missing, add it:

# Before VAE decode:
if hasattr(self.vae_config, 'latents_mean'):
    latents_mean = torch.tensor(self.vae_config.latents_mean).view(1, -1, 1, 1).to(latents)
    latents_std = torch.tensor(self.vae_config.latents_std).view(1, -1, 1, 1).to(latents)
    latents = latents * latents_std + latents_mean
```

## Issue 4: Glyph Mask Not Aligned

### Symptom
Glyph embeddings are extracted but mask doesn't align with flattened sequence.

### How to Check
```python
# In glm_image_before_denoising.py, after glyph encoding:
print(f"[DEBUG] Glyph encoding:")
print(f"  Number of glyphs: {len(texts_to_encode)}")
print(f"  Valid mask sum: {text_inputs.attention_mask.sum()}")
print(f"  Final prompt_embeds shape: {prompt_embeds.shape}")
print(f"  Attention mask shape: {attention_mask.shape if attention_mask is not None else None}")
```

### Potential Fix
```python
# Ensure mask matches flattened glyph length
if batch.do_classifier_free_guidance:
    # After concatenating embeddings
    # Verify mask length matches embedding length
    assert attention_mask.shape[1] == prompt_embeds.shape[1], \
        f"Mask length {attention_mask.shape[1]} != embedding length {prompt_embeds.shape[1]}"
```

## Testing Procedure

### Step 1: Add Debug Prints
```bash
# Add debug prints to key locations
# Run inference
python examples/inference/basic/basic_glm_image.py 2>&1 | tee debug.log

# Check output for:
# - Tensor shapes
# - Mask presence
# - Value ranges
```

### Step 2: Compare with sglang
```python
# If you have sglang code, run similar debug prints
# Compare:
# 1. Tensor shapes at each stage
# 2. Mask shapes and values
# 3. Latent value ranges
```

### Step 3: Targeted Fixes
```bash
# Make ONE fix at a time
# Test SSIM after each fix
pytest fastvideo/tests/ssim/test_glm_image_similarity.py -v -s

# Expected progression:
# 0.66 → 0.75-0.80 (attention mask)
# 0.75-0.80 → 0.85-0.92 (VAE normalization)
```

## Common Pitfalls

### 1. Mask Dimension Mismatch
```python
# Wrong: mask is [B, S] but attention expects [B, H, S_q, S_k]
# Fix: Reshape or expand mask appropriately
```

### 2. Mask Not Propagated
```python
# Wrong: mask created in before_denoising but not passed to transformer
# Fix: Add mask to forward() signatures and pass through
```

### 3. Normalization Order Wrong
```python
# Wrong: normalize after other transforms
# Fix: Check sglang order - normalize might need to be first/last
```

### 4. Per-Channel vs Global Normalization
```python
# Wrong: Using global mean/std
# Fix: Use per-channel if that's what sglang does
```

## Quick Wins to Try

### Try 1: Pass Mask Through
```python
# In GlmImageTransformer2DModel.forward():
# Add attention_mask parameter
# Pass to transformer blocks
# Pass to attention layers

# Expected SSIM gain: +0.05 to +0.15
```

### Try 2: Check Mask Format
```python
# Print mask in attention layer
# Verify it's not all ones (useless mask)
# Verify it's not inverted (0=keep, 1=mask would be wrong)
```

### Try 3: Latent Stats Check
```python
# Print latent statistics
# Compare with known good values
# If out of range, add normalization
```

## Expected SSIM After Each Fix

| Fix | Expected SSIM | Reason |
|-----|---------------|--------|
| Baseline (dim=2) | 0.66 | Current |
| + Mask passed | 0.70-0.75 | Better token handling |
| + Mask shaped | 0.75-0.80 | Correct masking |
| + VAE norm | 0.85-0.92 | Proper latent space |

## Summary

The key is to:
1. **Verify tensor shapes** - understand the actual layout
2. **Track mask flow** - ensure it reaches attention
3. **Check VAE stats** - latents in correct range
4. **Test incrementally** - one fix at a time

With systematic debugging, you should be able to identify and fix the remaining gaps to reach 0.9+ SSIM.
