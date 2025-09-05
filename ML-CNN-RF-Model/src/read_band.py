import rasterio
from rasterio.enums import Resampling
import numpy as np

def read_band(path, out_shape=None, ref_profile=None):
    """
    Read a Sentinel-2 band, optionally resample to match a reference shape/profile.
    Returns: numpy array, band profile
    """
    with rasterio.open(path) as src:
        if out_shape and ref_profile:
            # Resample to reference shape
            data = src.read(
                1,
                out_shape=out_shape,
                resampling=Resampling.bilinear
            )
            profile = src.profile
            profile.update({
                'height': out_shape[0],
                'width': out_shape[1],
                'transform': ref_profile['transform']
            })
            return data, profile
        else:
            data = src.read(1)
            return data, src.profile
