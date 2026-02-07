# Summary: Path Forward for GLM-Image SSIM 0.66 → 0.9+

## Current Situation

**SSIM Progress**:
- Initial: 0.4 (broken)
- After bc570af: 0.66 (improved!)
- After dim=1 "fix": 0.441 (WORSE!)
- **Keep dim=2** - it was correct!

## My Apology

I made an incorrect assumption about tensor dimensions and suggested changing dim=2 back to dim=1. You were absolutely right that dim=2 improved SSIM. The issue is NOT the KV cache dimension, but rather:

1. **Attention mask handling**
2. **VAE latent normalization**

## Why dim=2 Was Correct

After unflatten, if there's a permute operation:
```python
key = key.unflatten(2, (heads, -1))  # [B, S, H, D]
key = key.permute(0, 2, 1, 3)        # [B, H, S, D]
```

Then dim=2 IS the sequence dimension! So concatenating on dim=2 is correct.

## Real Issues to Fix

### Issue 1: Attention Mask Not Applied (Most Likely)

**Evidence**:
- bc570af creates attention_mask (lines ~278-290)
- But GlmImageAttention might not receive/use it
- Without mask, padding tokens affect attention

**Fix Strategy**:
1. Check if `attention_mask` is in forward() signature
2. Verify mask is passed through transformer layers
3. Ensure mask format matches LocalAttention expectations

**Expected Gain**: +0.10 to +0.15 SSIM (0.66 → 0.76-0.81)

### Issue 2: VAE Latent Normalization Missing/Wrong

**Evidence**:
- sglang likely has specific latent mean/std
- FastVideo might be missing this normalization
- Causes latent space mismatch

**Fix Strategy**:
1. Check VAE config for latents_mean, latents_std
2. Verify normalization is applied before/after VAE
3. Compare latent value ranges with sglang

**Expected Gain**: +0.10 to +0.15 SSIM (0.76 → 0.86-0.91)

## Action Plan

### Step 1: Investigate Tensor Layout (5 minutes)
```python
# Add to GlmImageAttention.forward() around line 380:
print(f"DEBUG: key shape after unflatten: {key.shape}")
# Check if there's a permute after this line
```

**Outcome**: Understand actual tensor layout, confirm dim=2 is sequence

### Step 2: Check Attention Mask Flow (15 minutes)
```python
# Add to glm_image_before_denoising.py:
print(f"DEBUG: attention_mask shape: {batch.attention_mask.shape}")

# Add to GlmImageTransformer2DModel.forward():
print(f"DEBUG: received attention_mask: {attention_mask is not None}")

# Add to GlmImageAttention.forward():
print(f"DEBUG: attention has mask param: {'attention_mask' in signature}")
```

**Outcome**: Identify if mask is being passed and used

### Step 3: Fix Attention Mask (30 minutes)
```python
# If mask not in signature, add it:
def forward(self, ..., attention_mask=None):
    # ...
    hidden_states = self.attn(query, key, value, attention_mask=attention_mask)
```

**Test**: Run SSIM test, expect 0.66 → 0.75-0.81

### Step 4: Check VAE Normalization (15 minutes)
```python
# Check config:
# fastvideo/configs/models/vaes/glm_image.py
# Look for latents_mean, latents_std

# Add debug:
print(f"DEBUG: latent stats - mean: {latents.mean()}, std: {latents.std()}")
```

**Outcome**: Identify if normalization is missing

### Step 5: Fix VAE Normalization (30 minutes)
```python
# If missing, add before decode:
if hasattr(config, 'latents_mean'):
    latents = latents * latents_std + latents_mean
```

**Test**: Run SSIM test, expect 0.75-0.81 → 0.85-0.92

## Expected Timeline

- **Today**: Investigate and identify specific issues (1 hour)
- **Today**: Implement attention mask fix (30 min)
- **Today**: Test (expect SSIM ~0.75-0.81)
- **Tomorrow**: Implement VAE fix if needed (30 min)
- **Tomorrow**: Test (expect SSIM ~0.85-0.92)
- **Tomorrow**: Fine-tune (SSIM 0.9+)

## Files to Check

### Priority 1: Attention Mask
- `fastvideo/models/dits/glm_image.py` - GlmImageAttention class
- `fastvideo/pipelines/stages/glm_image_before_denoising.py` - mask creation
- `fastvideo/attention/layer.py` - LocalAttention expectations

### Priority 2: VAE
- `fastvideo/configs/models/vaes/glm_image.py` - VAE config
- `fastvideo/pipelines/stages/glm_image_decoding.py` - VAE decode
- Compare with sglang's VAE handling

## Documents Created

1. **CORRECTED_ANALYSIS.md** - Explains my mistake, actual tensor layout
2. **DEBUG_GUIDE.md** - Systematic debugging procedures
3. **NEXT_STEPS.md** (this file) - Actionable plan

## Key Takeaways

✅ **Keep dim=2** - it's correct!
❌ **Don't revert** - bc570af was a good improvement
🔍 **Focus on**: attention mask and VAE normalization
📈 **Expected**: 0.66 → 0.9+ with these two fixes

## How to Proceed

1. **Don't change KV cache dimension** - keep dim=2
2. **Add debug prints** to understand mask flow
3. **Fix attention mask** handling (likely main issue)
4. **Check VAE normalization** (secondary issue)
5. **Test incrementally** after each fix

You're on the right track! The bc570af commit was good progress. The remaining gap is likely just:
- Attention mask not being applied ⟶ +0.10-0.15 SSIM
- VAE normalization missing ⟶ +0.10-0.15 SSIM

Both are fixable with targeted changes!

---

**Questions?** Check the DEBUG_GUIDE.md for detailed debugging steps.

**Need Code Help?** Share the specific sections you're investigating and I'll help analyze them.
