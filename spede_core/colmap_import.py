"""Turn a COLMAP reconstruction (e.g. Mip-NeRF 360) into SpEdE camera packs.

Camera packs hold square pinhole views, so every image is centre-cropped to a
square and its principal point shifted accordingly. Poses stay in the COLMAP
world, which is the frame 3DGS models trained on that capture use.

A pack is an NPZ with ``targets`` [N,H,W,3] uint8, ``labels``, ``viewmats``
[N,4,4] (world → camera), ``intrinsics`` [N,3,3], ``frame_indices`` and, for
compatibility with earlier packs, ``rgb_weights``, ``is_source`` and
``face_size``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .imageio import rgb_u8


@dataclass(frozen=True)
class ColmapView:
    name: str
    viewmat: np.ndarray
    intrinsic: np.ndarray
    image_path: Path


def read_colmap_views(sparse_dir: Path, images_dir: Path) -> list[ColmapView]:
    """PINHOLE/SIMPLE_PINHOLE views, intrinsics rescaled to the image files
    actually present (COLMAP may have run on another resolution)."""
    try:
        import pycolmap
    except ImportError as error:          # optional dependency: only this importer needs it
        raise ImportError("reading a COLMAP reconstruction needs pycolmap: pip install pycolmap") from error

    model = pycolmap.Reconstruction(str(sparse_dir))
    views = []
    for image in sorted(model.images.values(), key=lambda item: item.name):
        camera = model.cameras[image.camera_id]
        path = Path(images_dir) / image.name
        if not path.exists():
            # Downscaled Mip-NeRF 360 folders keep the names but may change case.
            matches = [p for p in Path(images_dir).iterdir() if p.name.lower() == image.name.lower()]
            if not matches:
                raise FileNotFoundError(f"image for {image.name} not found in {images_dir}")
            path = matches[0]
        with Image.open(path) as handle:
            width, height = handle.size
        k = np.asarray(camera.calibration_matrix(), np.float64)
        k[0] *= width / float(camera.width)
        k[1] *= height / float(camera.height)
        world_to_camera = np.eye(4)
        world_to_camera[:3, :] = image.cam_from_world().matrix()
        views.append(ColmapView(image.name, world_to_camera.astype(np.float32), k.astype(np.float32), path))
    return views


def square_crop(view: ColmapView, size: int | None) -> tuple[np.ndarray, np.ndarray]:
    """The view's photo centre-cropped to a square (resized to ``size`` when
    given) and its intrinsics moved with the crop."""
    with Image.open(view.image_path) as handle:
        image = np.asarray(handle.convert("RGB"))
    height, width = image.shape[:2]
    side = min(width, height)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    crop = image[y0:y0 + side, x0:x0 + side]
    k = np.asarray(view.intrinsic, np.float64).copy()
    k[0, 2] -= x0
    k[1, 2] -= y0
    if size is not None and size != side:
        scale = size / float(side)
        crop = np.asarray(Image.fromarray(crop).resize((size, size), Image.Resampling.LANCZOS))
        k[:2] *= scale
    return crop, k.astype(np.float32)


def write_camera_pack(views: list[ColmapView], path: Path, *, size: int | None = None) -> Path:
    """A camera pack of the real photos (every view is a capture view)."""
    if not views:
        raise ValueError("at least one view is required")
    images, intrinsics = [], []
    for view in views:
        image, k = square_crop(view, size)
        images.append(rgb_u8(image))
        intrinsics.append(k)
    if len({im.shape for im in images}) > 1:
        raise ValueError("all views in a pack must share one resolution: pass --size")
    stack = np.stack(images)
    h, w = stack.shape[1:3]
    payload = dict(
        targets=stack,
        rgb_weights=np.ones((len(views), h, w), np.float16),
        viewmats=np.stack([np.asarray(v.viewmat, np.float32) for v in views]),
        intrinsics=np.stack(intrinsics),
        frame_indices=np.arange(len(views), dtype=np.int32),
        is_source=np.ones(len(views), bool),
        labels=np.asarray([Path(v.name).stem for v in views]),
        face_size=np.asarray(h, np.int32),
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def split_heldout(
    views: list[ColmapView],
    *,
    every: int = 8,
    heldout_count: int = 8,
    edit_count: int = 24,
    hero: str | None = None,
) -> tuple[list[ColmapView], list[ColmapView]]:
    """The Mip-NeRF 360 convention: every ``every``-th image is a test image.

    Returns (edit views, held-out views), each subsampled evenly along the
    capture order so both packs cover the whole orbit. The hero, if given, is
    always an edit view and never held out.
    """
    test = [v for i, v in enumerate(views) if i % every == 0 and v.name != hero]
    train = [v for i, v in enumerate(views) if i % every != 0 and v.name != hero]

    def spread(items: list[ColmapView], count: int) -> list[ColmapView]:
        if count >= len(items):
            return list(items)
        picks = np.linspace(0, len(items) - 1, count).round().astype(int)
        return [items[i] for i in picks]

    edit = spread(train, edit_count - (1 if hero else 0))
    if hero:
        hero_view = next((v for v in views if v.name == hero), None)
        if hero_view is None:
            raise ValueError(f"hero image {hero!r} is not in the reconstruction")
        edit = [hero_view, *edit]
    return edit, spread(test, heldout_count)
