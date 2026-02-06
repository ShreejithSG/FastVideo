# Numerical Alignment Tests

This directory contains component-level numerical alignment tests for FastVideo.

## Purpose

These tests verify that FastVideo's optimized implementations produce numerically identical (or very close) outputs compared to reference implementations (typically from diffusers or transformers libraries).

## Test Categories

### 1. Component Tests
- **VAE Tests**: Encode/decode alignment
- **Text Encoder Tests**: Embedding alignment
- **Transformer/DiT Tests**: Forward pass alignment
- **Scheduler Tests**: Timestep and noise scheduling alignment
- **Attention Tests**: Various attention mechanism implementations

### 2. Tolerance Levels

Different components have different numerical precision requirements:

- **Strict** (atol=1e-4, rtol=1e-4): VAE encoding, Text encoding
- **Moderate** (atol=1e-2, rtol=1e-2): Transformer forward passes
- **Relaxed** (atol=1e-1, rtol=1e-1): Full denoising loops with many steps

### 3. Running Tests

Run all alignment tests:
```bash
pytest fastvideo/tests/alignment/ -v
```

Run specific component tests:
```bash
pytest fastvideo/tests/alignment/test_vae_alignment.py -v
pytest fastvideo/tests/alignment/test_transformer_alignment.py -v
```

## Expected SSIM Ranges

For end-to-end video generation tests:

| Precision | Expected SSIM | Acceptable Range |
|-----------|---------------|------------------|
| fp32 | 0.98-1.0 | > 0.95 |
| bf16 | 0.93-0.98 | > 0.90 |
| fp16 | 0.90-0.95 | > 0.85 |

If SSIM falls below these ranges, investigate component-level alignment.

## Troubleshooting Low SSIM

If you're seeing SSIM < 0.5:

1. **Run component tests** to isolate the issue
2. **Check precision settings** - mixed precision can cause drift
3. **Verify model weights** - ensure models are loaded correctly
4. **Check random seeds** - must be identical for both reference and test
5. **Inspect attention backends** - different backends have different numerical properties
6. **Review optimizations** - some optimizations trade precision for speed

## Adding New Alignment Tests

Use this template:

```python
import torch
from torch.testing import assert_close

def test_component_alignment():
    # Load reference implementation
    ref_model = ReferenceModel.from_pretrained(...)
    
    # Load FastVideo implementation
    fv_model = FastVideoModel.load(...)
    
    # Create identical inputs with fixed seed
    torch.manual_seed(42)
    inputs = create_test_inputs()
    
    # Run both models
    with torch.no_grad():
        ref_output = ref_model(**inputs)
        fv_output = fv_model(**inputs)
    
    # Compare outputs with appropriate tolerance
    assert_close(ref_output, fv_output, atol=1e-4, rtol=1e-4)
```
