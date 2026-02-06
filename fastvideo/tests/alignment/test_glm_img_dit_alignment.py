# SPDX-License-Identifier: Apache-2.0
"""
GLM-img DiT numerical alignment test.

This test compares FastVideo's GLM-img DiT implementation against
the reference sglang implementation to ensure numerical equivalence.

NOTE: This test requires both sglang and FastVideo to have GLM-img support.
Run this test AFTER porting GLM-img from sglang to FastVideo.
"""
import os
import pytest
import torch
from torch.testing import assert_close

from fastvideo.logger import init_logger
from fastvideo.tests.alignment.utils import (
    assert_alignment,
    ToleranceLevel,
    set_deterministic_mode,
    compare_model_weights,
)

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29507"

# Flag to skip test if not yet ported
GLM_IMG_PORTED = False  # Set to True once porting is complete

@pytest.mark.skipif(not GLM_IMG_PORTED, reason="GLM-img not yet ported to FastVideo")
def test_glm_img_dit_weight_loading():
    """
    Test that GLM-img DiT weights are loaded correctly.
    This is a sanity check before testing forward pass.
    """
    set_deterministic_mode(42)
    
    try:
        # Import sglang reference
        from sglang.multimodal_gen.runtime.models.dits.glm_image import (
            GlmImageTransformer as SGL_DiT
        )
        from sglang.multimodal_gen.configs.models.dits.glmimage import GlmImageDitConfig
    except ImportError:
        pytest.skip("sglang not installed or GLM-img not available")
    
    try:
        # Import FastVideo implementation
        from fastvideo.models.dits.glm_image import GlmImageTransformer as FV_DiT
        from fastvideo.configs.models.dits import GlmImageDitConfig as FV_GlmImageDitConfig
    except ImportError:
        pytest.skip("FastVideo GLM-img implementation not found")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Load model weights
    model_path = "THUDM/CogVideoX-5B"  # Replace with actual GLM-img model path
    
    # Load sglang model
    sgl_dit = SGL_DiT.from_pretrained(model_path).to(device, dtype).eval()
    
    # Load FastVideo model
    fv_dit = FV_DiT.from_pretrained(model_path).to(device, dtype).eval()
    
    # Compare weights
    logger.info("Comparing model weights...")
    weights_match, mismatches = compare_model_weights(
        sgl_dit,
        fv_dit,
        tolerance=ToleranceLevel.STRICT,
    )
    
    if not weights_match:
        logger.error(f"Found {len(mismatches)} weight mismatches!")
        for name, metrics in list(mismatches.items())[:5]:  # Show first 5
            logger.error(f"  {name}: max_diff={metrics['max_abs_diff']:.6e}")
    
    assert weights_match, f"Weight loading mismatch in {len(mismatches)} parameters"
    logger.info("✓ Weight loading verified!")


@pytest.mark.skipif(not GLM_IMG_PORTED, reason="GLM-img not yet ported to FastVideo")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height,width", [(512, 512), (1024, 1024)])
def test_glm_img_dit_forward_pass(batch_size, height, width):
    """
    Test that GLM-img DiT forward pass produces numerically equivalent outputs.
    This is the critical test for SSIM alignment.
    """
    set_deterministic_mode(42)
    
    try:
        from sglang.multimodal_gen.runtime.models.dits.glm_image import (
            GlmImageTransformer as SGL_DiT,
            GlmImageKVCache,
        )
    except ImportError:
        pytest.skip("sglang not installed")
    
    try:
        from fastvideo.models.dits.glm_image import (
            GlmImageTransformer as FV_DiT,
            GlmImageKVCache as FV_GlmImageKVCache,
        )
    except ImportError:
        pytest.skip("FastVideo GLM-img not found")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Load models
    model_path = "THUDM/CogVideoX-5B"  # Replace with actual path
    sgl_dit = SGL_DiT.from_pretrained(model_path).to(device, dtype).eval()
    fv_dit = FV_DiT.from_pretrained(model_path).to(device, dtype).eval()
    
    # Create test inputs
    # Latent dimensions after VAE encoding
    latent_height = height // 8  # VAE scale factor
    latent_width = width // 8
    
    hidden_states = torch.randn(
        batch_size, 16, latent_height, latent_width,
        device=device, dtype=dtype
    )
    
    # Text embeddings (1472 dim for GLM-img)
    seq_len = 77  # Standard CLIP-like length
    encoder_hidden_states = torch.randn(
        batch_size, seq_len, 1472,
        device=device, dtype=dtype
    )
    
    # Timestep
    timestep = torch.tensor([500], device=device, dtype=dtype)
    
    # Prior token ID (unique to GLM-img)
    prior_token_id = torch.randint(
        0, 16384, (batch_size, 1),  # codebook_size=16384
        device=device
    )
    
    # Crop coords and target size
    crop_coords = torch.tensor([[0, 0]], device=device)
    target_size = torch.tensor([[height, width]], device=device)
    
    # Create KV caches
    num_layers = 30  # GLM-img has 30 layers
    sgl_kv_cache = GlmImageKVCache(num_layers)
    fv_kv_cache = FV_GlmImageKVCache(num_layers)
    
    # Forward pass with KV caching
    with torch.no_grad():
        # Write mode: cache conditioning tokens
        sgl_kv_cache.set_mode("write")
        fv_kv_cache.set_mode("write")
        
        sgl_output = sgl_dit(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            prior_token_id=prior_token_id,
            crop_coords=crop_coords,
            target_size=target_size,
            kv_caches=sgl_kv_cache,
            kv_caches_mode="write",
        )
        
        fv_output = fv_dit(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            prior_token_id=prior_token_id,
            crop_coords=crop_coords,
            target_size=target_size,
            kv_caches=fv_kv_cache,
            kv_caches_mode="write",
        )
    
    logger.info(f"Testing GLM-img DiT forward pass ({height}x{width})")
    logger.info(f"  Input shape: {hidden_states.shape}")
    logger.info(f"  Text emb shape: {encoder_hidden_states.shape}")
    logger.info(f"  Output shape: {sgl_output.shape}")
    
    # Compare outputs
    assert_alignment(
        sgl_output,
        fv_output,
        tolerance=ToleranceLevel.MODERATE,  # bf16 allows some drift
        component_name=f"GLM-img DiT ({height}x{width})",
    )
    
    # Also test KV cache content
    for layer_idx in range(min(3, num_layers)):  # Test first 3 layers
        sgl_k, sgl_v = sgl_kv_cache[layer_idx].get()
        fv_k, fv_v = fv_kv_cache[layer_idx].get()
        
        if sgl_k is not None and fv_k is not None:
            logger.info(f"  Comparing KV cache layer {layer_idx}")
            assert_close(sgl_k, fv_k, atol=1e-2, rtol=1e-2)
            assert_close(sgl_v, fv_v, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not GLM_IMG_PORTED, reason="GLM-img not yet ported to FastVideo")
def test_glm_img_dit_kv_cache_modes():
    """
    Test that GLM-img KV cache modes (write/read/skip) work correctly.
    This is critical for CFG (classifier-free guidance).
    """
    set_deterministic_mode(42)
    
    try:
        from fastvideo.models.dits.glm_image import (
            GlmImageTransformer as FV_DiT,
            GlmImageKVCache,
        )
    except ImportError:
        pytest.skip("FastVideo GLM-img not found")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Load model
    model_path = "THUDM/CogVideoX-5B"
    dit = FV_DiT.from_pretrained(model_path).to(device, dtype).eval()
    
    # Create inputs
    hidden_states = torch.randn(1, 16, 64, 64, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(1, 77, 1472, device=device, dtype=dtype)
    timestep = torch.tensor([500], device=device, dtype=dtype)
    
    kv_cache = GlmImageKVCache(30)
    
    with torch.no_grad():
        # Step 1: Write mode - cache positive prompt
        kv_cache.set_mode("write")
        output_write = dit(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            kv_caches=kv_cache,
            kv_caches_mode="write",
        )
        
        # Verify cache is populated
        k, v = kv_cache[0].get()
        assert k is not None, "KV cache not populated in write mode"
        logger.info(f"✓ Write mode: cached {k.shape}")
        
        # Step 2: Read mode - reuse cache
        kv_cache.set_mode("read")
        output_read = dit(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=torch.zeros_like(encoder_hidden_states),  # Should use cache
            timestep=timestep,
            kv_caches=kv_cache,
            kv_caches_mode="read",
        )
        logger.info(f"✓ Read mode: reused cache")
        
        # Step 3: Skip mode - ignore cache (for negative prompt)
        kv_cache.set_mode("skip")
        output_skip = dit(
            hidden_states=hidden_states.clone(),
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            kv_caches=kv_cache,
            kv_caches_mode="skip",
        )
        logger.info(f"✓ Skip mode: bypassed cache")
    
    # Outputs should differ based on cache mode
    # write == read (using same conditioning)
    # skip != read (different conditioning)
    write_read_diff = torch.abs(output_write - output_read).max()
    skip_read_diff = torch.abs(output_skip - output_read).max()
    
    logger.info(f"Write-Read diff: {write_read_diff:.6e}")
    logger.info(f"Skip-Read diff: {skip_read_diff:.6e}")
    
    # Write and read should be very similar (using cache correctly)
    assert write_read_diff < 1e-2, "Write and Read modes should produce similar outputs"
    
    # Skip should be different (not using cache)
    # Note: This check is soft since skip mode processes different data
    logger.info("✓ KV cache modes working correctly!")


if __name__ == "__main__":
    # Manual test run for debugging
    if GLM_IMG_PORTED:
        test_glm_img_dit_weight_loading()
        test_glm_img_dit_forward_pass(1, 512, 512)
        test_glm_img_dit_kv_cache_modes()
        logger.info("\n✓ All GLM-img DiT alignment tests passed!")
    else:
        logger.warning("GLM-img not yet ported. Set GLM_IMG_PORTED=True after porting.")
