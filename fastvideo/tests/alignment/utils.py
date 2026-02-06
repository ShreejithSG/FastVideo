# SPDX-License-Identifier: Apache-2.0
"""
Utilities for numerical alignment testing.
"""
import torch
from torch.testing import assert_close
from typing import Dict, Any, Tuple, Optional
import numpy as np
from fastvideo.logger import init_logger

logger = init_logger(__name__)


class ToleranceLevel:
    """Standard tolerance levels for different component types."""
    STRICT = {"atol": 1e-4, "rtol": 1e-4}  # VAE, Text encoders
    MODERATE = {"atol": 1e-2, "rtol": 1e-2}  # Transformers
    RELAXED = {"atol": 1e-1, "rtol": 1e-1}  # Full pipelines


def compute_alignment_metrics(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Dict[str, float]:
    """
    Compute various alignment metrics between two tensors.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        
    Returns:
        Dictionary containing alignment metrics
    """
    # Ensure tensors have same shape
    assert tensor1.shape == tensor2.shape, f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
    
    # Convert to float32 for metrics computation
    t1 = tensor1.float()
    t2 = tensor2.float()
    
    # Absolute difference
    abs_diff = torch.abs(t1 - t2)
    
    # Relative difference (avoid division by zero)
    epsilon = 1e-8
    rel_diff = abs_diff / (torch.abs(t1) + epsilon)
    
    metrics = {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "std_abs_diff": abs_diff.std().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "cosine_similarity": torch.nn.functional.cosine_similarity(
            t1.flatten(), t2.flatten(), dim=0
        ).item(),
        "pearson_correlation": torch.corrcoef(
            torch.stack([t1.flatten(), t2.flatten()])
        )[0, 1].item(),
    }
    
    return metrics


def assert_alignment(
    output1: torch.Tensor,
    output2: torch.Tensor,
    tolerance: Dict[str, float],
    component_name: str = "Component",
    verbose: bool = True,
) -> None:
    """
    Assert that two tensors are aligned within tolerance.
    
    Args:
        output1: First tensor (reference)
        output2: Second tensor (test)
        tolerance: Dict with 'atol' and 'rtol' keys
        component_name: Name of component being tested (for logging)
        verbose: Whether to log detailed metrics
    """
    metrics = compute_alignment_metrics(output1, output2)
    
    if verbose:
        logger.info(f"\n{component_name} Alignment Metrics:")
        logger.info(f"  Max Abs Diff: {metrics['max_abs_diff']:.6e}")
        logger.info(f"  Mean Abs Diff: {metrics['mean_abs_diff']:.6e}")
        logger.info(f"  Max Rel Diff: {metrics['max_rel_diff']:.6e}")
        logger.info(f"  Mean Rel Diff: {metrics['mean_rel_diff']:.6e}")
        logger.info(f"  Cosine Similarity: {metrics['cosine_similarity']:.6f}")
        logger.info(f"  Pearson Correlation: {metrics['pearson_correlation']:.6f}")
    
    # Main assertion
    try:
        assert_close(output1, output2, **tolerance)
        logger.info(f"✓ {component_name} alignment passed!")
    except AssertionError as e:
        logger.error(f"✗ {component_name} alignment failed!")
        logger.error(f"  Tolerance: atol={tolerance['atol']}, rtol={tolerance['rtol']}")
        logger.error(f"  Max violation: {metrics['max_abs_diff']:.6e}")
        raise


def set_deterministic_mode(seed: int = 42) -> None:
    """
    Set deterministic mode for reproducible testing.
    
    Args:
        seed: Random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Enable deterministic algorithms (may be slower)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compare_model_weights(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    tolerance: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare weights of two models.
    
    Args:
        model1: First model (reference)
        model2: Second model (test)
        tolerance: Tolerance for weight comparison (default: STRICT)
        
    Returns:
        Tuple of (all_match, mismatch_info)
    """
    if tolerance is None:
        tolerance = ToleranceLevel.STRICT
    
    params1 = dict(model1.named_parameters())
    params2 = dict(model2.named_parameters())
    
    mismatches = {}
    all_match = True
    
    # Check if same parameters exist
    keys1 = set(params1.keys())
    keys2 = set(params2.keys())
    
    if keys1 != keys2:
        logger.warning(f"Parameter name mismatch!")
        logger.warning(f"  Only in model1: {keys1 - keys2}")
        logger.warning(f"  Only in model2: {keys2 - keys1}")
        all_match = False
    
    # Compare common parameters
    for name in keys1 & keys2:
        p1 = params1[name]
        p2 = params2[name]
        
        # Handle DTensor if needed
        if hasattr(p2, 'to_local'):
            p2 = p2.to_local()
        
        try:
            assert_close(p1, p2, **tolerance)
        except AssertionError:
            metrics = compute_alignment_metrics(p1, p2)
            mismatches[name] = metrics
            all_match = False
            logger.warning(f"Weight mismatch in {name}:")
            logger.warning(f"  Max diff: {metrics['max_abs_diff']:.6e}")
    
    return all_match, mismatches


def estimate_expected_ssim(
    precision: str,
    num_denoising_steps: int,
    attention_backend: str = "FLASH_ATTN",
) -> Tuple[float, float]:
    """
    Estimate expected SSIM range based on configuration.
    
    Args:
        precision: Precision string (fp32, bf16, fp16)
        num_denoising_steps: Number of denoising steps
        attention_backend: Attention backend being used
        
    Returns:
        Tuple of (min_expected_ssim, max_expected_ssim)
    """
    # Base SSIM by precision
    base_ranges = {
        "fp32": (0.98, 1.0),
        "bf16": (0.93, 0.98),
        "fp16": (0.90, 0.95),
    }
    
    min_ssim, max_ssim = base_ranges.get(precision, (0.85, 0.95))
    
    # Adjust for number of steps (more steps = more accumulation)
    step_penalty = min(0.02 * (num_denoising_steps // 10), 0.05)
    min_ssim = max(0.85, min_ssim - step_penalty)
    
    # Adjust for attention backend
    if attention_backend != "FLASH_ATTN":
        # Different backends may have slight variations
        min_ssim = max(0.85, min_ssim - 0.01)
    
    return min_ssim, max_ssim


def diagnose_ssim_failure(
    ssim_value: float,
    expected_range: Tuple[float, float],
    component_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    Provide diagnostic information for SSIM failures.
    
    Args:
        ssim_value: Measured SSIM value
        expected_range: Expected (min, max) SSIM range
        component_metrics: Optional component-level metrics
        
    Returns:
        Diagnostic message string
    """
    min_expected, max_expected = expected_range
    
    diagnosis = f"\n{'='*60}\n"
    diagnosis += f"SSIM Diagnostic Report\n"
    diagnosis += f"{'='*60}\n"
    diagnosis += f"Measured SSIM: {ssim_value:.4f}\n"
    diagnosis += f"Expected Range: [{min_expected:.4f}, {max_expected:.4f}]\n"
    
    if ssim_value < min_expected:
        gap = min_expected - ssim_value
        diagnosis += f"\n⚠️  SSIM is {gap:.4f} below expected minimum!\n"
        
        if ssim_value < 0.5:
            diagnosis += "\n🔴 CRITICAL: SSIM < 0.5 indicates major alignment issues!\n"
            diagnosis += "\nLikely causes:\n"
            diagnosis += "  1. Model weights not loaded correctly\n"
            diagnosis += "  2. Different random seeds between reference and test\n"
            diagnosis += "  3. Major numerical precision issues\n"
            diagnosis += "  4. Incorrect model configuration\n"
        elif ssim_value < 0.8:
            diagnosis += "\n🟠 WARNING: SSIM < 0.8 indicates significant drift!\n"
            diagnosis += "\nLikely causes:\n"
            diagnosis += "  1. Accumulation of numerical errors in denoising loop\n"
            diagnosis += "  2. Attention mechanism differences\n"
            diagnosis += "  3. VAE encoding/decoding precision issues\n"
            diagnosis += "  4. Scheduler implementation differences\n"
        else:
            diagnosis += "\n🟡 MODERATE: SSIM slightly below expected.\n"
            diagnosis += "\nLikely causes:\n"
            diagnosis += "  1. Minor numerical precision differences\n"
            diagnosis += "  2. Different random number generation\n"
            diagnosis += "  3. Optimization flags affecting computation order\n"
    
    diagnosis += "\nRecommended actions:\n"
    diagnosis += "  1. Run component-level alignment tests\n"
    diagnosis += "  2. Check all random seeds are identical\n"
    diagnosis += "  3. Verify precision settings (fp32/bf16/fp16)\n"
    diagnosis += "  4. Compare intermediate outputs step-by-step\n"
    
    if component_metrics:
        diagnosis += "\n\nComponent Metrics:\n"
        for component, metrics in component_metrics.items():
            diagnosis += f"\n{component}:\n"
            for metric, value in metrics.items():
                diagnosis += f"  {metric}: {value}\n"
    
    diagnosis += f"\n{'='*60}\n"
    
    return diagnosis
