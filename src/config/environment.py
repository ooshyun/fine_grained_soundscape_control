"""Environment configuration for CUDA and PyTorch debugging."""

import os
import logging
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EnvironmentConfig:
    """
    CUDA and PyTorch environment configuration.

    These environment variables enable detailed debugging and error tracking
    for distributed training scenarios.

    Attributes:
        torch_distributed_debug: Level of distributed training debug info
        torch_use_cuda_dsa: Enable device-side assertions
        cuda_launch_blocking: Make CUDA kernel calls synchronous for debugging
        debug_distributed_training: Enable custom distributed training debug
    """

    torch_distributed_debug: str = "DETAIL"
    torch_use_cuda_dsa: str = "1"
    cuda_launch_blocking: str = "1"
    debug_distributed_training: str = "1"

    def to_env_dict(self) -> Dict[str, str]:
        """
        Convert configuration to environment variable dictionary.

        Returns:
            Dictionary mapping environment variable names to values
        """
        return {
            "TORCH_DISTRIBUTED_DEBUG": self.torch_distributed_debug,
            "TORCH_USE_CUDA_DSA": self.torch_use_cuda_dsa,
            "CUDA_LAUNCH_BLOCKING": self.cuda_launch_blocking,
            "DEBUG_DISTRIBUTUTED_TRAINING": self.debug_distributed_training,
        }

    def apply(self) -> None:
        """Apply environment variables to os.environ."""
        for key, value in self.to_env_dict().items():
            os.environ[key] = value


def setup_cuda_environment(
    config: EnvironmentConfig = None,
) -> EnvironmentConfig:
    """
    Set up CUDA and PyTorch environment variables for debugging.

    This function configures the environment for better error tracking during
    distributed training. Note that these settings may impact performance
    and should primarily be used during development and debugging.

    Performance Impact:
        - CUDA_LAUNCH_BLOCKING=1: Makes CUDA calls synchronous (slower)
        - TORCH_USE_CUDA_DSA: Adds device-side assertion overhead

    Args:
        config: Environment configuration (uses defaults if None)

    Returns:
        Applied EnvironmentConfig instance

    Note:
        CUDA kernel errors are normally reported asynchronously, which makes
        debugging difficult. CUDA_LAUNCH_BLOCKING=1 forces synchronous execution
        so errors appear at their actual source location.
    """
    if config is None:
        config = EnvironmentConfig()

    logging.info("Setting up CUDA environment variables...")
    config.apply()

    logging.info("Environment variables configured:")
    for key, value in config.to_env_dict().items():
        logging.info(f"  {key}: {value}")

    return config


def get_production_environment() -> EnvironmentConfig:
    """
    Get environment configuration optimized for production.

    Disables synchronous execution for better performance.

    Returns:
        EnvironmentConfig with production settings
    """
    return EnvironmentConfig(
        torch_distributed_debug="OFF",
        torch_use_cuda_dsa="0",
        cuda_launch_blocking="0",
        debug_distributed_training="0",
    )


def get_debug_environment() -> EnvironmentConfig:
    """
    Get environment configuration optimized for debugging.

    Enables all debugging features (default).

    Returns:
        EnvironmentConfig with debug settings
    """
    return EnvironmentConfig()  # Uses defaults
