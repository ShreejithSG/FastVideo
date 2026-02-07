# SPDX-License-Identifier: Apache-2.0
"""
Scheduler numerical alignment tests.

Tests verify that FastVideo's scheduler implementations produce identical
timesteps and noise schedules compared to diffusers reference implementations.
"""
import os
import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch.testing import assert_close

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.loader.component_loader import SchedulerLoader
from fastvideo.configs.models.schedulers import FlowMatchEulerConfig
from fastvideo.tests.alignment.utils import (
    assert_alignment,
    ToleranceLevel,
    set_deterministic_mode,
)

logger = init_logger(__name__)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29505"


@pytest.mark.parametrize("num_steps", [6, 20, 50])
@pytest.mark.parametrize("shift", [1.0, 7.0, 17.0])
def test_flow_match_scheduler_timesteps(num_steps, shift):
    """
    Test that FastVideo's FlowMatch scheduler produces identical timesteps
    to the reference diffusers implementation.
    """
    set_deterministic_mode(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create reference scheduler
    ref_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    ref_scheduler.set_timesteps(num_steps, device=device)
    ref_timesteps = ref_scheduler.timesteps
    
    # Create FastVideo scheduler
    args = FastVideoArgs(
        pipeline_config=PipelineConfig(
            scheduler_config=FlowMatchEulerConfig(shift=shift)
        )
    )
    loader = SchedulerLoader()
    fv_scheduler = loader.load(scheduler_config=FlowMatchEulerConfig(shift=shift), args=args)
    fv_scheduler.set_timesteps(num_steps, device=device)
    fv_timesteps = fv_scheduler.timesteps
    
    logger.info(f"Testing scheduler with {num_steps} steps, shift={shift}")
    logger.info(f"Reference timesteps: {ref_timesteps[:5]}...")
    logger.info(f"FastVideo timesteps: {fv_timesteps[:5]}...")
    
    # Timesteps should be exactly identical
    assert_alignment(
        ref_timesteps,
        fv_timesteps,
        tolerance=ToleranceLevel.STRICT,
        component_name=f"FlowMatch Scheduler (steps={num_steps}, shift={shift})",
    )


@pytest.mark.parametrize("num_steps", [6, 20, 50])
@pytest.mark.parametrize("shift", [7.0, 17.0])
def test_flow_match_scheduler_step(num_steps, shift):
    """
    Test that FastVideo's FlowMatch scheduler step produces identical
    outputs to the reference implementation.
    """
    set_deterministic_mode(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    
    # Create schedulers
    ref_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    ref_scheduler.set_timesteps(num_steps, device=device)
    
    args = FastVideoArgs(
        pipeline_config=PipelineConfig(
            scheduler_config=FlowMatchEulerConfig(shift=shift)
        )
    )
    loader = SchedulerLoader()
    fv_scheduler = loader.load(scheduler_config=FlowMatchEulerConfig(shift=shift), args=args)
    fv_scheduler.set_timesteps(num_steps, device=device)
    
    # Create sample input (latent noise)
    batch_size = 1
    channels = 16
    num_frames = 21
    height = 90
    width = 160
    
    sample = torch.randn(
        batch_size, channels, num_frames, height, width,
        device=device, dtype=dtype
    )
    
    # Create model output (predicted noise)
    model_output = torch.randn_like(sample)
    
    # Test scheduler step for each timestep
    for i, timestep in enumerate(ref_scheduler.timesteps[:3]):  # Test first 3 steps
        # Reference scheduler step
        ref_result = ref_scheduler.step(
            model_output.clone(),
            timestep,
            sample.clone(),
            return_dict=False,
        )[0]
        
        # FastVideo scheduler step
        fv_result = fv_scheduler.step(
            model_output.clone(),
            fv_scheduler.timesteps[i],
            sample.clone(),
            return_dict=False,
        )[0]
        
        logger.info(f"Testing step {i+1}/{num_steps} (timestep={timestep.item():.3f})")
        
        # Results should be very close (fp32 precision)
        assert_alignment(
            ref_result,
            fv_result,
            tolerance=ToleranceLevel.STRICT,
            component_name=f"Scheduler Step {i+1}",
        )


def test_scheduler_state_consistency():
    """
    Test that scheduler state remains consistent across multiple steps.
    """
    set_deterministic_mode(42)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    num_steps = 20
    shift = 7.0
    
    # Create scheduler
    args = FastVideoArgs(
        pipeline_config=PipelineConfig(
            scheduler_config=FlowMatchEulerConfig(shift=shift)
        )
    )
    loader = SchedulerLoader()
    scheduler = loader.load(scheduler_config=FlowMatchEulerConfig(shift=shift), args=args)
    scheduler.set_timesteps(num_steps, device=device)
    
    # Initial state
    initial_timesteps = scheduler.timesteps.clone()
    
    # Run a dummy step
    sample = torch.randn(1, 16, 21, 90, 160, device=device, dtype=dtype)
    model_output = torch.randn_like(sample)
    
    _ = scheduler.step(model_output, scheduler.timesteps[0], sample)
    
    # Timesteps should remain unchanged
    assert torch.equal(scheduler.timesteps, initial_timesteps), \
        "Scheduler timesteps changed after step!"
    
    logger.info("✓ Scheduler state consistency verified")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_flow_match_scheduler_timesteps(6, 7.0)
    test_flow_match_scheduler_timesteps(20, 17.0)
    test_flow_match_scheduler_step(6, 7.0)
    test_scheduler_state_consistency()
    logger.info("\n✓ All scheduler alignment tests passed!")
