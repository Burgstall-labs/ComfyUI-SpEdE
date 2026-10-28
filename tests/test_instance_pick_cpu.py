"""CPU tests: each camera keeps the SAM instances that hold the hero's object.

Synthetic scene: two boxes of Gaussian centres side by side; the hero mask is
on box A, and the text search masks box B (the larger one) in other cameras.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spede_assets import instances_from_comfy_batch, write_segment_report
from spede_core.selection import FAILED_NAME, _fail, collapse_message, pick_hero_views
from spede_core.visual_hull import (
    HullView, hero_surface_ids, mask_instances, pick_hero_instances, project_points,
)

SIZE = 128
K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])


def viewmat(shift_x: float = 0.0) -> np.ndarray:
    """A camera looking down +z from (shift_x, 0, 0)."""
    m = np.eye(4)
    m[0, 3] = -shift_x
    return m


def box(x0, x1, y0, y1, z0, z1, step=0.02):
    xs, ys, zs = (np.arange(a, b + 1e-9, step) for a, b in ((x0, x1), (y0, y1), (z0, z1)))
    return np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), -1).reshape(-1, 3)


# Box A (the clicked object) and a larger box B beside it, both at z 5-5.4,
# and a back wall at z 10.
A = box(-1.4, -0.6, -0.4, 0.4, 5.0, 5.4, 0.04)
B = box(0.4, 1.6, -0.6, 0.6, 5.0, 5.4, 0.04)
WALL = box(-3.0, 3.0, -3.0, 3.0, 10.0, 10.0, 0.05)
MEANS = np.concatenate([A, B, WALL])
IS_A = np.r_[np.ones(len(A), bool), np.zeros(len(B) + len(WALL), bool)]


def render(points: np.ndarray, vm: np.ndarray):
    """Nearest-centre depth and alpha (a stand-in for the renderer's depth)."""
    x, y, ok = project_points(points, vm, K, SIZE, SIZE)
    z = points @ vm[2, :3] + vm[2, 3]
    depth = np.full((SIZE, SIZE), np.inf)
    np.minimum.at(depth, (y[ok], x[ok]), z[ok])
    alpha = np.isfinite(depth).astype(np.float32)
    return np.where(np.isfinite(depth), depth, 0.0), alpha


def silhouette(points: np.ndarray, vm: np.ndarray) -> np.ndarray:
    x, y, ok = project_points(points, vm, K, SIZE, SIZE)
    mask = np.zeros((SIZE, SIZE), bool)
    mask[y[ok].min():y[ok].max() + 1, x[ok].min():x[ok].max() + 1] = True
    return mask


def hero_view() -> HullView:
    vm = viewmat()
    depth, alpha = render(MEANS, vm)
    return HullView("hero", vm, K, silhouette(A, vm), depth, alpha)


def other_view(name: str, shift: float, mask: np.ndarray | None = None) -> HullView:
    vm = viewmat(shift)
    depth, alpha = render(MEANS, vm)
    return HullView(name, vm, K, silhouette(B, vm) if mask is None else mask, depth, alpha)


def test_hero_surface_is_the_object_not_the_wall_behind_it():
    hero = hero_view()
    ids = hero_surface_ids(MEANS, hero.viewmat, hero.intrinsic, hero.silhouette, hero.depth, hero.alpha)
    assert len(ids) > 50 and IS_A[ids].all()
    # Without depth the wall behind A in the mask's frustum comes along.
    loose = hero_surface_ids(MEANS, hero.viewmat, hero.intrinsic, hero.silhouette, None, None)
    assert (~IS_A[loose]).any()


def test_the_instance_with_the_hero_object_wins_over_a_larger_wrong_one():
    hero = hero_view()
    points = MEANS[hero_surface_ids(MEANS, hero.viewmat, hero.intrinsic, hero.silhouette,
                                    hero.depth, hero.alpha)]
    view = other_view("cam", 0.3)
    wrong, right = silhouette(B, view.viewmat), silhouette(A, view.viewmat)
    assert wrong.sum() > right.sum()
    chosen, info = pick_hero_instances(points, view, np.stack([wrong, right]), occlusion_margin=1.0)
    assert info["chosen"] == [1] and info["fractions"][0] < 0.1 < info["fractions"][1]
    assert np.array_equal(chosen, right)


def test_no_instance_on_the_hero_object_leaves_the_view_empty():
    hero = hero_view()
    points = MEANS[hero_surface_ids(MEANS, hero.viewmat, hero.intrinsic, hero.silhouette,
                                    hero.depth, hero.alpha)]
    view = other_view("cam", -0.3)
    chosen, info = pick_hero_instances(points, view, silhouette(B, view.viewmat)[None], occlusion_margin=1.0)
    assert not chosen.any() and info["chosen"] == [] and info["reason"] == "no instance holds the object"
    # The object out of frame: nothing to judge by, the view is empty too.
    away = HullView("away", viewmat(40.0), K, view.silhouette)
    chosen, info = pick_hero_instances(points, away, view.silhouette[None])
    assert not chosen.any() and info["reason"] == "object not in view"


def test_union_masks_fall_back_to_their_connected_pieces():
    vm = viewmat(0.2)
    union = silhouette(A, vm) | silhouette(B, vm)
    pieces = mask_instances(union)
    assert len(pieces) == 2 and np.array_equal(pieces.any(axis=0), union)
    assert len(mask_instances(np.zeros((4, 4), bool))) == 0


def test_pick_hero_views_keeps_sam_instances_and_pieces_and_drops_wrong_views(tmp_path):
    hero = hero_view()
    sam = other_view("sam", 0.3)
    npz = tmp_path / "instances.npz"
    np.savez_compressed(npz, instances=np.stack([silhouette(B, sam.viewmat), silhouette(A, sam.viewmat)]))
    wrong = other_view("wrong", -0.3)                       # union of B only
    both = other_view("both", 0.1, silhouette(A, viewmat(0.1)) | silhouette(B, viewmat(0.1)))
    report = {"hero_view_id": "hero", "views": {"sam": {"instances_path": str(npz), "instances": 2},
                                                "wrong": {}, "both": {}}}
    kept, info, surface = pick_hero_views(report, MEANS, hero, [sam, wrong, both])
    assert [v.view_id for v in kept] == ["sam", "both"] and info["dropped"] == ["wrong"]
    assert info["views"]["sam"]["source"] == "sam" and info["views"]["both"]["source"] == "pieces"
    assert np.array_equal(kept[0].silhouette, silhouette(A, sam.viewmat))
    assert np.array_equal(kept[1].silhouette, silhouette(A, viewmat(0.1)))
    assert IS_A[surface].all()


def test_comfy_instances_split_per_view_and_land_in_the_report(tmp_path):
    labels = ["hero", "c1", "c2", "c3"]
    cams = {v: SimpleNamespace(width=8, height=8, is_virtual=False, synthetic=False) for v in labels}
    batch = np.zeros((5, 8, 8), np.float32)
    for i in range(5):
        batch[i, i, :] = 1.0
    boxes = [[{"x": 0}], [{"x": 0}, {"x": 1}], [], [{"x": 0}, {"x": 1}]]
    found = instances_from_comfy_batch(batch, boxes, labels, cams)
    assert [len(found[v]) for v in labels] == [1, 2, 0, 2]
    with pytest.raises(ValueError, match="boxes count"):
        instances_from_comfy_batch(batch[:4], boxes, labels, cams)
    masks = {v: found[v].any(axis=0) for v in labels}
    masks["c2"] = np.ones((8, 8), bool)
    _path, _hero, report = write_segment_report(cams, masks, hero_view_id="hero", output_dir=tmp_path,
                                                instances={v: found[v] for v in labels if v != "c2"},
                                                object_text="bread in plastic wrap")
    assert report["object_text"] == "bread in plastic wrap"
    assert "instances_path" not in report["views"]["hero"] and "instances_path" not in report["views"]["c2"]
    with np.load(report["views"]["c3"]["instances_path"]) as f:
        assert np.array_equal(f["instances"], found["c3"] > 0.5)


def test_collapsed_selection_fails_loudly_and_the_ui_explains_it(tmp_path):
    message = collapse_message("bread in plastic wrap", ["no camera could judge the removal"], 21, 23)
    assert message.startswith("Selection collapsed: no camera could judge the removal.")
    assert '"bread in plastic wrap" finds other objects' in message and "21 of 23 cameras" in message
    assert "'" not in message                             # ComfyUI quotes errors with single quotes
    selection = tmp_path / "job/selection"
    selection.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Selection collapsed"):
        _fail(selection, message, judged=0)
    assert json.loads((selection / FAILED_NAME).read_text())["judged"] == 0
    studio = next(p for p in (Path(__file__).resolve().parents[1] / "competition" / "studio",
                              Path(__file__).resolve().parents[1] / "studio") if p.is_dir())
    sys.path.insert(0, str(studio))
    import comfy_backend

    assert comfy_backend.selection_message(tmp_path / "job") == message
    assert comfy_backend.selection_message(tmp_path / "other") is None
