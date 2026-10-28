#!/usr/bin/env python3
"""Geometry helpers of the SpEdE pack, each printing one JSON object.

``rank-views`` and ``pick-camera`` render with gsplat (CUDA); the UI runs them
as subprocesses so its server never holds the GPU itself. ``import-colmap``
is CPU only and builds the camera packs a scene needs.

    python geometry_cli.py rank-views --source-ply scene.ply --camera-targets cameras.npz \
        --hero-view-id DSCF5565 --hero-mask hero_mask.png
    python geometry_cli.py pick-camera --source-ply scene.ply --camera-targets cameras.npz \
        --pose-json pose.json [--render-out view.png]
    python geometry_cli.py import-colmap --sparse-dir scene/sparse/0 --images-dir scene/images_4 \
        --out-dir scene/cameras [--size 512 --hero DSCF5565 --edit-count 24 --heldout-count 8 --test-every 8]

``pose.json`` is a three.js camera and a click: ``position``, ``quaternion``
(x, y, z, w), ``fov`` (vertical, degrees), ``aspect``, ``x`` and ``y`` (click,
0..1 of the viewer size).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402


def _renderer(args):
    from spede_core.renderer import CameraSet, GsplatRenderer

    cameras = CameraSet.from_camera_pack(Path(args.camera_targets))
    return GsplatRenderer(source_ply=Path(args.source_ply), cameras=cameras, device=args.device), cameras


def rank_views_command(args) -> dict:
    from PIL import Image

    from spede_core.view_rank import rank_views

    renderer, cameras = _renderer(args)
    if args.hero_view_id not in cameras.by_id:
        raise SystemExit(f"hero camera {args.hero_view_id!r} is not in {args.camera_targets}")
    mask = np.asarray(Image.open(args.hero_mask).convert("L")) > 127
    return rank_views(renderer, cameras, args.hero_view_id, mask)


def pick_camera_command(args) -> dict:
    from spede_core.view_rank import pick_hero, three_camera_to_view

    pose = json.loads(Path(args.pose_json).read_text())
    view = three_camera_to_view(pose["position"], pose["quaternion"], float(pose["fov"]), float(pose["aspect"]))
    renderer, cameras = _renderer(args)
    pixel = (float(pose["x"]) * view.width, float(pose["y"]) * view.height)
    result = pick_hero(renderer, cameras, view, pixel)
    if args.render_out is not None and result.get("status") == "picked":
        from spede_core.imageio import save_rgb

        out = Path(args.render_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        save_rgb(out, renderer.render(result["view_id"], edited=False))
        result["render"] = str(out)
    return result


def import_colmap_command(args) -> dict:
    """Edit and held-out camera packs from a COLMAP reconstruction (CPU)."""
    from spede_core.colmap_import import read_colmap_views, split_heldout, write_camera_pack

    views = read_colmap_views(Path(args.sparse_dir), Path(args.images_dir))
    edit, heldout = split_heldout(views, every=args.test_every, heldout_count=args.heldout_count,
                                  edit_count=args.edit_count, hero=args.hero)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "edit_cameras": str(write_camera_pack(edit, out / "edit_cameras.npz", size=args.size)),
        "heldout_cameras": str(write_camera_pack(heldout, out / "heldout_cameras.npz", size=args.size)),
        "edit_views": [Path(v.name).stem for v in edit],
        "heldout_views": [Path(v.name).stem for v in heldout],
        "hero": Path(args.hero).stem if args.hero else Path(edit[0].name).stem,
        "size": args.size,
    }
    (out / "camera_split.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (("rank-views", rank_views_command), ("pick-camera", pick_camera_command)):
        p = sub.add_parser(name)
        p.add_argument("--source-ply", required=True)
        p.add_argument("--camera-targets", required=True)
        p.add_argument("--device", default="cuda")
        p.set_defaults(func=func)
        if name == "rank-views":
            p.add_argument("--hero-view-id", required=True)
            p.add_argument("--hero-mask", required=True)
        else:
            p.add_argument("--pose-json", required=True)
            p.add_argument("--render-out", default=None)
    colmap = sub.add_parser("import-colmap", help="edit and held-out camera packs from a COLMAP reconstruction")
    colmap.add_argument("--sparse-dir", required=True, help="COLMAP model folder (cameras/images/points3D)")
    colmap.add_argument("--images-dir", required=True, help="the photos the model refers to (any downscale)")
    colmap.add_argument("--out-dir", required=True)
    colmap.add_argument("--size", type=int, default=512, help="square crop size in pixels")
    colmap.add_argument("--hero", default=None, help="image name (with extension) that must be an edit view")
    colmap.add_argument("--edit-count", type=int, default=24)
    colmap.add_argument("--heldout-count", type=int, default=8)
    colmap.add_argument("--test-every", type=int, default=8, help="every n-th image is held out (Mip-NeRF 360: 8)")
    colmap.set_defaults(func=import_colmap_command)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(args.func(args), indent=2))


if __name__ == "__main__":
    main()
