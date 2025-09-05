import numpy as np
import rasterio
from rasterio.enums import Resampling


def load_scl_mask(scl_path: str, target_shape=None):
	"""
	Load Sentinel-2 Scene Classification Layer (SCL) and build a boolean clear-sky mask.

	Args:
		scl_path: Path to SCL JP2 (usually 20m)
		target_shape: Optional (height, width) to resample mask to

	Returns:
		clear_mask: np.ndarray[bool] where True indicates clear pixels
	"""
	with rasterio.open(scl_path) as src:
		if target_shape is not None:
			scl = src.read(1, out_shape=target_shape, resampling=Resampling.nearest)
		else:
			scl = src.read(1)

	# SCL cloud/snow classes to exclude: 8,9,10,11
	cloudy = (scl >= 8) & (scl <= 11)
	return ~cloudy


def apply_clear_mask(array: np.ndarray, clear_mask: np.ndarray) -> np.ndarray:
	"""
	Apply clear-sky mask to a raster array, setting cloudy pixels to NaN.
	"""
	if clear_mask is None:
		return array
	masked = array.astype(float).copy()
	masked[~clear_mask] = np.nan
	return masked

#!/usr/bin/env python3
"""
Cloud masking utilities for Sentinel-2 L2A (SCL band) and s2cloudless hook.

Usage in JP2 rasterio workflows or to document masking policy when using GEE.
"""

from typing import Optional, Tuple
import numpy as np

try:
    # Optional dependency; we only provide a hook here
    from s2cloudless import S2PixelCloudDetector  # type: ignore
    S2CLOUDLESS_AVAILABLE = True
except Exception:
    S2CLOUDLESS_AVAILABLE = False


def mask_with_scl(scl_band: np.ndarray, valid_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Create a boolean mask using Sentinel-2 Scene Classification (SCL) codes.

    Cloud/Cirrus/Snow codes masked out (False):
      3: Cloud shadows, 8: Medium probability clouds,
      9: High probability clouds, 10: Thin cirrus, 11: Snow/ice

    Vegetation/Soil/Water kept (True).

    Args:
        scl_band: 2D ndarray with SCL values (uint8)
        valid_shape: optional shape to assert against

    Returns:
        mask (bool ndarray): True == keep pixel; False == mask out
    """
    if valid_shape is not None and scl_band.shape != valid_shape:
        raise ValueError(f"SCL shape {scl_band.shape} != expected {valid_shape}")

    scl = scl_band.astype(np.uint8)
    mask = np.ones_like(scl, dtype=bool)
    # Mask out unwanted SCL classes
    for bad_code in (3, 8, 9, 10, 11):
        mask &= scl != bad_code
    return mask


def s2cloudless_mask(rgbn: np.ndarray, probability_threshold: float = 0.4) -> np.ndarray:
    """
    Optional s2cloudless mask. Expects stacked bands in order [B2, B3, B4, B8].
    Returns True for clear pixels.
    """
    if not S2CLOUDLESS_AVAILABLE:
        raise RuntimeError("s2cloudless not installed. Install with: pip install s2cloudless")

    if rgbn.ndim != 3 or rgbn.shape[2] != 4:
        raise ValueError("rgbn must be HxWx4 with [B2,B3,B4,B8]")

    detector = S2PixelCloudDetector(threshold=probability_threshold, average_over=4, dilation_size=2)
    # s2cloudless expects reflectance 0..1
    x = np.clip(rgbn.astype(np.float32), 0.0, 1.0)
    cloud_prob = detector.get_cloud_probability_maps(x)
    cloud_mask = cloud_prob >= (probability_threshold * 100.0)
    return ~cloud_mask


