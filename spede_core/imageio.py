from __future__ import annotations

from pathlib import Path
import numpy as np
from PIL import Image


def rgb_u8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("RGB image must have shape [H, W, 3]")
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("RGB image contains non-finite values")
        if float(array.max(initial=0.0)) <= 1.0 + 1e-6:
            array = np.rint(np.clip(array, 0.0, 1.0) * 255.0)
        else:
            array = np.rint(np.clip(array, 0.0, 255.0))
    return array.astype(np.uint8, copy=False)


def mask_u8(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError("mask must have shape [H, W] or [H, W, 1]")
    if np.issubdtype(array.dtype, np.floating):
        threshold = 0.5 if float(array.max(initial=0.0)) <= 1.0 + 1e-6 else 127.5
        array = array >= threshold
    elif array.dtype != np.bool_:
        array = array > 0
    return array.astype(np.uint8) * 255


def save_rgb(path: Path, image: np.ndarray, *, compress_level: int = 6) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_u8(image), mode="RGB").save(path, compress_level=int(compress_level))
    return path


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_mask(path: Path, mask: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8(mask), mode="L").save(path)
    return path


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) > 127
