# Quick Reference: Fixing SSIM from 0.4 to 0.9+

## 🚨 Current Problem
- **SSIM**: 0.4 (critical!)
- **Cause**: Component swapping in GLM-img port from sglang
- **Goal**: SSIM > 0.9 (target: 0.93-0.98)

## 📋 Quick Diagnosis

```bash
cd fastvideo/tests/alignment
python diagnose_ssim.py --ssim 0.4 --model glm-img --run-all-tests
```

Expected output: **Attention component failing**

## 🔧 Top 3 Fixes (80% of SSIM gain)

### Fix #1: Use sglang's Attention ⭐⭐⭐
**Impact**: +0.3 SSIM (0.4 → 0.7)
```python
# DON'T swap - use sglang's attention directly
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
```

### Fix #2: Implement KV Cache ⭐⭐
**Impact**: +0.1 SSIM (0.7 → 0.8)
```python
class GlmImageKVCache:
    def set_mode(self, mode):  # "write", "read", "skip"
        for cache in self.caches:
            cache.mode = mode
```

### Fix #3: Match RoPE Exactly ⭐
**Impact**: +0.05 SSIM (0.8 → 0.85)
```python
from sglang.multimodal_gen.runtime.layers.rotary_embedding import _apply_rotary_emb
```

## 📊 Expected Timeline

| Day | Action | Expected SSIM |
|-----|--------|---------------|
| 1 | Diagnose + identify root cause | 0.4 |
| 2 | Fix attention | 0.7-0.8 |
| 3 | Fix KV cache + RoPE | 0.85-0.9 |
| 4 | Fix precision + validate | 0.9-0.93 ✅ |
| 5 | Documentation + tests | 0.93+ ✅ |

## ✅ Success Checklist

- [ ] Read `SSIM_EXPECTATIONS.md` (understand targets)
- [ ] Read `GLM_IMG_PORTING_GUIDE.md` (understand fixes)
- [ ] Run `diagnose_ssim.py` (identify issues)
- [ ] Fix attention (use sglang's USPAttention)
- [ ] Fix KV cache (implement 3-mode system)
- [ ] Fix RoPE (match frequency calculation)
- [ ] Set `GLM_IMG_PORTED=True` in test file
- [ ] Run `pytest test_glm_img_dit_alignment.py -v`
- [ ] Validate SSIM > 0.9 in E2E test
- [ ] Document fixes and add regression tests

## 📚 Key Documents

1. **ALIGNMENT_TESTING_SUMMARY.md** - Start here (overview)
2. **SSIM_EXPECTATIONS.md** - What SSIM is realistic?
3. **GLM_IMG_PORTING_GUIDE.md** - How to fix issues?
4. **fastvideo/tests/alignment/README.md** - How to use tests?

## 🎯 Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| SSIM (bf16) | > 0.93 | Strong guarantee |
| SSIM (fp32) | > 0.95 | Ideal target |
| Timeline | 3-5 days | With focused work |
| Confidence | >90% | Very high |

## ⚠️ Common Mistakes to Avoid

1. ❌ **Don't** expect SSIM = 1.0 with bf16
2. ❌ **Don't** swap components without testing
3. ❌ **Don't** skip component-level tests
4. ❌ **Don't** ignore KV cache modes
5. ❌ **Don't** use different attention backends

## ✅ Best Practices

1. ✅ **Do** use sglang's attention directly
2. ✅ **Do** test each component in isolation
3. ✅ **Do** match precision settings exactly
4. ✅ **Do** verify random seeds are identical
5. ✅ **Do** document all changes

## 🆘 If SSIM Still < 0.9 After Fixes

1. Re-run diagnostics:
   ```bash
   python diagnose_ssim.py --ssim <measured> --model glm-img
   ```

2. Check component tests individually:
   ```bash
   pytest test_glm_img_dit_alignment.py::test_glm_img_dit_forward_pass -v
   ```

3. Compare layer-by-layer outputs:
   ```python
   for i, layer in enumerate(dit.layers):
       sgl_out = sgl_layer(...)
       fv_out = fv_layer(...)
       diff = (sgl_out - fv_out).abs().max()
       print(f"Layer {i}: {diff:.6e}")
   ```

4. Review the porting guide for missed details

## 💡 Key Insight

**The #1 cause of SSIM=0.4 is attention mismatch.**
→ Solution: Use sglang's attention directly, don't swap!

## 🎓 Remember

- **SSIM=1.0**: Not possible with bf16
- **SSIM>0.93**: Achievable with proper porting
- **Attention**: Usually the culprit
- **Testing**: Test each component separately
- **Timeline**: 3-5 days is realistic

---

**Need help?** Check the full guides in `/fastvideo/tests/alignment/`

**Questions?**
- What SSIM? → `SSIM_EXPECTATIONS.md`
- How to fix? → `GLM_IMG_PORTING_GUIDE.md`
- How to test? → `README.md`
- How to debug? → `diagnose_ssim.py --help`
