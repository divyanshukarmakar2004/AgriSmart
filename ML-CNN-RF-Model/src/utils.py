"""
Utility helpers for file management and JP2 → PNG preview/conversion.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

try:
    import imageio  # type: ignore
    _HAS_IMAGEIO = True
except Exception:
    _HAS_IMAGEIO = False
try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0..255 uint8 for visualization.

    Handles raw reflectance (0..10000) and float (0..1) inputs.
    """
    data = arr.astype(np.float32)
    # If values look like Sentinel reflectance scaled 0..10000
    if data.max() > 2.0:
        data = data / 10000.0
    # Min-max scale if mostly constant range
    dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)
    data = np.clip(data, 0.0, 1.0)
    return (data * 255.0).round().astype(np.uint8)


def process_jp2(file_path: str, output_dir: str = 'Outputs/', img_dir: str = 'IMG_DATA/') -> Optional[str]:
    """Move a JP2 to IMG_DATA/ (if needed), convert first band to PNG in Outputs/.

    Args:
        file_path: Path to source .jp2 (can be in project root)
        output_dir: Directory to save PNG
        img_dir: Directory where JP2 files should live

    Returns:
        Path to generated PNG (str) or None if failed
    """
    src = Path(file_path)
    if not src.exists():
        print(f"File not found: {src}")
        return None

    img_root = Path(img_dir)
    out_root = Path(output_dir)
    img_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # Move file if it is not inside img_root already
    if img_root.resolve() not in src.resolve().parents:
        dest = img_root / src.name
        try:
            shutil.move(str(src), str(dest))
            moved_path = dest
            print(f"Moved {src.name} → {moved_path}")
        except Exception as move_err:
            print(f"Move failed for {src}: {move_err}")
            return None
    else:
        moved_path = src

    # Read first band and normalize
    try:
        with rasterio.open(moved_path) as ds:
            band1 = ds.read(1, resampling=Resampling.nearest)
            png_arr = _normalize_to_uint8(band1)
    except Exception as read_err:
        print(f"Read failed for {moved_path}: {read_err}")
        return None

    # Save PNG
    png_path = out_root / (moved_path.stem + '.png')
    try:
        if _HAS_IMAGEIO:
            imageio.imwrite(png_path.as_posix(), png_arr)
        elif _HAS_PIL:
            Image.fromarray(png_arr).save(png_path.as_posix())
        else:
            # Fallback to NumPy .npy if no writers available
            np.save(png_path.with_suffix('.npy').as_posix(), png_arr)
            png_path = png_path.with_suffix('.npy')
        print(f"Saved PNG: {png_path}")
        return str(png_path)
    except Exception as write_err:
        print(f"Write failed for {png_path}: {write_err}")
        return None

import os

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
