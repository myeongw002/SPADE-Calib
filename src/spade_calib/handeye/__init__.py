"""Full-recording trajectory Hand-Eye and uncertainty estimates.

Reported covariance is always marked with the assumed noise model and
is not independent physical calibration accuracy.
"""
from .solver import HandEyeResult, HandEyeSettings, fit_full_trajectory, fit_motion_pairs
from .uncertainty import analyze_local_uncertainty, block_bootstrap_uncertainty

__all__ = [
    "HandEyeResult", "HandEyeSettings", "fit_full_trajectory", "fit_motion_pairs",
    "analyze_local_uncertainty", "block_bootstrap_uncertainty",
]