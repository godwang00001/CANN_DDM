"""Core support modules for the CANN/DDM rate model."""

from .calibration import calibrate_target_diffusion_profile
from .default_params import build_stable_default_params

__all__ = ["build_stable_default_params", "calibrate_target_diffusion_profile"]
