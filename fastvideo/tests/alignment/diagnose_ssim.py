#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SSIM Diagnostic Tool

This tool helps diagnose low SSIM scores by running component-level
alignment tests and identifying where numerical drift occurs.

Usage:
    python diagnose_ssim.py --model Wan2.1-T2V-1.3B-Diffusers --ssim 0.4
    python diagnose_ssim.py --run-all-tests
"""
import argparse
import sys
import os
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastvideo.logger import init_logger
from fastvideo.tests.alignment.utils import (
    diagnose_ssim_failure,
    estimate_expected_ssim,
)

logger = init_logger(__name__)


def run_component_tests(model_name: str = "Wan2.1-T2V-1.3B-Diffusers"):
    """Run all component-level alignment tests for a model."""
    import subprocess
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running Component Tests for {model_name}")
    logger.info(f"{'='*60}\n")
    
    tests = [
        ("VAE", "fastvideo/tests/vaes/test_wan_vae.py"),
        ("Text Encoder", "fastvideo/tests/encoders/test_t5_encoder.py"),
        ("Transformer", "fastvideo/tests/transformers/test_wanvideo.py"),
        ("Scheduler", "fastvideo/tests/alignment/test_scheduler_alignment.py"),
    ]
    
    results = {}
    
    for component_name, test_path in tests:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Testing {component_name}...")
        logger.info(f"{'─'*60}")
        
        if not os.path.exists(test_path):
            logger.warning(f"Test file not found: {test_path}")
            results[component_name] = "SKIP"
            continue
        
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            if result.returncode == 0:
                logger.info(f"✓ {component_name} test PASSED")
                results[component_name] = "PASS"
            else:
                logger.error(f"✗ {component_name} test FAILED")
                logger.error(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
                results[component_name] = "FAIL"
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {component_name} test TIMEOUT")
            results[component_name] = "TIMEOUT"
        except Exception as e:
            logger.error(f"✗ {component_name} test ERROR: {e}")
            results[component_name] = "ERROR"
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Component Test Summary")
    logger.info(f"{'='*60}")
    
    for component, result in results.items():
        status_symbol = {
            "PASS": "✓",
            "FAIL": "✗",
            "SKIP": "⊘",
            "TIMEOUT": "⏱",
            "ERROR": "⚠",
        }
        symbol = status_symbol.get(result, "?")
        logger.info(f"{symbol} {component}: {result}")
    
    return results


def analyze_ssim(ssim_value: float, precision: str = "bf16", num_steps: int = 20):
    """Analyze a reported SSIM value and provide recommendations."""
    logger.info(f"\n{'='*60}")
    logger.info("SSIM Analysis")
    logger.info(f"{'='*60}\n")
    
    logger.info(f"Reported SSIM: {ssim_value:.4f}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Denoising steps: {num_steps}")
    
    # Estimate expected range
    expected_min, expected_max = estimate_expected_ssim(precision, num_steps)
    
    # Generate diagnosis
    diagnosis = diagnose_ssim_failure(
        ssim_value,
        (expected_min, expected_max),
        component_metrics=None,
    )
    
    logger.info(diagnosis)
    
    # Specific recommendations based on severity
    if ssim_value < 0.5:
        logger.info("\n🔴 CRITICAL ACTIONS REQUIRED:\n")
        logger.info("1. Verify model weights are loaded correctly")
        logger.info("   - Check model path")
        logger.info("   - Verify model version matches")
        logger.info("   - Ensure no cached corrupted weights")
        logger.info("\n2. Verify random seeds are set identically")
        logger.info("   - torch.manual_seed()")
        logger.info("   - torch.cuda.manual_seed_all()")
        logger.info("   - Generator seed parameter")
        logger.info("\n3. Run component tests to isolate failing component")
        logger.info("   python diagnose_ssim.py --run-all-tests")
    
    elif ssim_value < 0.8:
        logger.info("\n🟠 RECOMMENDED ACTIONS:\n")
        logger.info("1. Check precision settings match reference")
        logger.info("2. Verify scheduler configuration (shift, steps)")
        logger.info("3. Test VAE encode/decode separately")
        logger.info("4. Compare attention backend settings")
        logger.info("5. Run component alignment tests")
    
    else:
        logger.info("\n🟡 MINOR OPTIMIZATIONS:\n")
        logger.info("1. Consider using fp32 for higher precision")
        logger.info("2. Reduce number of denoising steps")
        logger.info("3. Fine-tune scheduler parameters")


def main():
    parser = argparse.ArgumentParser(description="SSIM Diagnostic Tool")
    parser.add_argument(
        "--ssim",
        type=float,
        help="Measured SSIM value to analyze"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Precision used in generation"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Wan2.1-T2V-1.3B-Diffusers",
        help="Model name"
    )
    parser.add_argument(
        "--run-all-tests",
        action="store_true",
        help="Run all component alignment tests"
    )
    
    args = parser.parse_args()
    
    logger.info("FastVideo SSIM Diagnostic Tool")
    logger.info("="*60)
    
    if args.run_all_tests:
        results = run_component_tests(args.model)
        
        # Check if any tests failed
        failures = [k for k, v in results.items() if v == "FAIL"]
        if failures:
            logger.error(f"\n❌ Component test failures detected: {', '.join(failures)}")
            logger.error("These components are likely causing SSIM degradation.")
            sys.exit(1)
        else:
            logger.info("\n✓ All component tests passed!")
            logger.info("SSIM issues may be due to accumulation across pipeline.")
    
    if args.ssim is not None:
        analyze_ssim(args.ssim, args.precision, args.steps)
    
    logger.info("\n" + "="*60)
    logger.info("Diagnostic complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
