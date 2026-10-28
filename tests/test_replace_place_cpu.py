"""CPU checks for Replace's placement (spede_core.replace_place): the support
plane, grounding, scale by silhouette area, the size modes, the shift cap, the
inverse-extract base with its footprint fill and the contact shadow."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from spede_core.ply import GaussianPly
from spede_core.replace_place import (
    TRIPOSPLAT_UP, ObjectGaussians, SupportPlane, contact_shadow, fill_rows, footprint_extents, keep_largest_component,
    load_object_ply, object_rows, place_object, point_render, quaternion_from_matrix, replace_fill,
    rotation_about_axis, support_plane, write_scene,
)

UP = np.array([0.0, 0.0, 1.0])


def look_at(eye, target, up=UP) -> np.ndarray:
    """World-to-camera (OpenCV: x right, y down, z forward)."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    view = np.eye(4)
    view[:3, :3] = np.stack([x, y, z])
    view[:3, 3] = -view[:3, :3] @ eye
    return view


K = np.array([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]])
VIEW = look_at([0.0, -3.0, 2.0], [0.0, 0.0, 0.3])
PLANE = SupportPlane(normal=UP.copy(), offset=0.0, footprint_centre=np.zeros(3), footprint_size=1.0,
                     object_height=0.6, spread=0.001, ring=1000, tilt_deg=0.0)


def box(n: int, size, centre=(0, 0, 0), seed: int = 0, colour=(0.8, 0.2, 0.2)) -> ObjectGaussians:
    """A solid box of small opaque Gaussians."""
    rng = np.random.default_rng(seed)
    means = (rng.random((n, 3)) - 0.5) * np.asarray(size, float) + np.asarray(centre, float)
    rgb = np.tile(colour, (n, 1)).astype(float)
    return ObjectGaussians(means, np.full((n, 3), math.log(0.01)), np.tile([1.0, 0, 0, 0], (n, 1)),
                           np.full(n, 5.0), rgb)


def generated_box(n: int = 6000) -> ObjectGaussians:
    """A box in TripoSplat's frame (up along -Y): 0.5 wide, 1.0 tall, 0.3 deep."""
    return box(n, (0.5, 1.0, 0.3), seed=1)


def mask_of(obj: ObjectGaussians) -> np.ndarray:
    return point_render(obj, VIEW, K, 256, 256)[1] > 0.5


def test_support_plane_fits_the_ring_around_the_removed_object():
    rng = np.random.default_rng(0)
    # A table at z = 0.2 tilted 2 degrees about x, an object standing on it at the origin.
    tilt = math.radians(2.0)
    table_uv = (rng.random((20000, 2)) - 0.5) * 4
    table = np.c_[table_uv, 0.2 + np.tan(tilt) * table_uv[:, 1] + rng.normal(0, 0.002, 20000)]
    obj = (rng.random((3000, 3)) - 0.5) * [0.6, 0.4, 0.8] + [0, 0, 0.6]
    means = np.vstack([table, obj])
    ids = np.arange(len(table), len(means))
    plane = support_plane(means, np.full(len(means), 4.0), ids, UP)
    assert abs(plane.tilt_deg - 2.0) < 0.3
    assert abs(plane.height(np.array([[0.0, 0.0, 0.2]]))[0]) < 0.01
    assert abs(plane.object_height - 0.8) < 0.05 and abs(plane.footprint_size - 0.6) < 0.05
    assert abs(plane.footprint_centre[2] - 0.2) < 0.01 and np.linalg.norm(plane.footprint_centre[:2]) < 0.02
    # Nothing round the object: no surface to stand on.
    with pytest.raises(ValueError, match="no surface"):
        support_plane(obj, np.full(len(obj), 4.0), np.arange(len(obj) // 2), UP)


def test_quaternions_follow_the_rotation():
    r = rotation_about_axis([0.3, -0.5, 0.8], 2.4)
    w, x, y, z = quaternion_from_matrix(r)
    back = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    assert np.allclose(back, r, atol=1e-9)


def test_place_object_grounds_on_the_plane_and_scales_by_silhouette_area():
    # Qwen drew a box 0.9 tall standing on the plane at the old footprint (scale 0.9 of the generated one).
    target = box(6000, (0.45, 0.27, 0.9), centre=(0.05, 0.0, 0.45), seed=2)
    placed, report = place_object(generated_box(), plane=PLANE, viewmat=VIEW, intrinsic=K, mask=mask_of(target),
                                  max_shift_fraction=0.5)
    height = PLANE.height(placed.means)
    assert abs(np.percentile(height, 1)) < 0.02                       # the base is on the plane
    assert abs(report["object_height"] - 0.9) < 0.06
    assert abs(report["scale"] - 0.9) < 0.06
    assert report["silhouette_iou"] > 0.85 and not report["shift_capped"]
    assert abs(report["base_centre"][0] - 0.05) < 0.04
    # Log-scales follow the uniform scale.
    assert np.allclose(placed.log_scales, math.log(0.01) + math.log(report["scale"]))


def test_place_object_caps_the_shift_from_the_old_footprint():
    target = box(6000, (0.5, 0.3, 1.0), centre=(0.6, 0.0, 0.5), seed=3)    # drawn 0.6 to the side
    placed, report = place_object(generated_box(), plane=PLANE, viewmat=VIEW, intrinsic=K, mask=mask_of(target),
                                  max_shift_fraction=0.2)
    assert report["shift_capped"] is True
    assert report["shift"] == pytest.approx(0.2, abs=1e-6)            # 0.2 of the 1.0 footprint
    assert report["shift_fraction"] == pytest.approx(0.2, abs=1e-6)
    assert abs(np.percentile(PLANE.height(placed.means), 1)) < 0.02
    free, report = place_object(generated_box(), plane=PLANE, viewmat=VIEW, intrinsic=K, mask=mask_of(target),
                                max_shift_fraction=1.0)
    assert not report["shift_capped"] and abs(report["base_centre"][0] - 0.6) < 0.05


def test_place_object_tells_front_from_back_by_colour():
    # Red front (towards -y in the scene after placement), blue back: same silhouette both ways round.
    gen = generated_box(8000)
    front = gen.means[:, 2] < 0                      # generated -z faces the camera at yaw 0
    rgb = np.where(front[:, None], [0.9, 0.1, 0.1], [0.1, 0.1, 0.9])
    gen = ObjectGaussians(gen.means, gen.log_scales, gen.quaternions, gen.opacity_logits, rgb)
    picture = np.zeros((256, 256, 3))
    picture[...] = [0.1, 0.1, 0.9]                   # Qwen drew the blue side towards the camera
    target = box(6000, (0.5, 0.3, 1.0), centre=(0, 0, 0.5), seed=4)
    _placed, blue = place_object(gen, plane=PLANE, viewmat=VIEW, intrinsic=K, mask=mask_of(target), image=picture)
    picture[...] = [0.9, 0.1, 0.1]
    _placed, red = place_object(gen, plane=PLANE, viewmat=VIEW, intrinsic=K, mask=mask_of(target), image=picture)
    turn = abs(((blue["yaw_deg"] - red["yaw_deg"]) + 180) % 360 - 180)
    assert turn > 150 and red["photometric_error"] < 0.1 and blue["photometric_error"] < 0.1


def test_size_modes_match_the_removed_objects_dominant_extent():
    # The removed object: 0.6 tall, a footprint 1.0 long and 0.5 wide: its dominant extent is sqrt(0.5) = 0.71.
    plane = SupportPlane(normal=UP.copy(), offset=0.0, footprint_centre=np.zeros(3), footprint_size=1.0,
                         object_height=0.6, spread=0.001, ring=1000, tilt_deg=0.0, footprint_major=1.0,
                         footprint_minor=0.5)
    target = box(6000, (0.45, 0.27, 0.9), centre=(0.05, 0.0, 0.45), seed=2)      # Qwen drew it 0.9 tall
    runs = {size: place_object(generated_box(), plane=plane, viewmat=VIEW, intrinsic=K, mask=mask_of(target),
                               max_shift_fraction=0.5, size=size)[1] for size in ("natural", "match", "larger")}
    # Natural: Qwen's size (the silhouette's area), as before.
    assert abs(runs["natural"]["object_height"] - 0.9) < 0.06 and runs["natural"]["scale_vs_natural"] == 1.0
    # Match: the tall box's height (its dominant extent) becomes the old footprint's width, whatever Qwen drew.
    assert plane.dominant_extent == pytest.approx(math.sqrt(0.5))
    assert runs["match"]["dominant_vs_removed"] == pytest.approx(1.0, abs=1e-6)
    assert runs["match"]["object_height"] == pytest.approx(math.sqrt(0.5), abs=0.01)
    assert runs["match"]["height_natural_vs_removed"] == pytest.approx(runs["natural"]["height_vs_removed"], rel=0.02)
    assert runs["match"]["footprint_dominant"] is False and runs["match"]["cover_scale"] is None
    # Larger: 1.3 times that; the base stays on the plane, the position near Qwen's.
    assert runs["larger"]["scale"] == pytest.approx(1.3 * runs["match"]["scale"], rel=1e-6)
    for report in runs.values():
        assert abs(report["base_centre"][2]) < 0.02 and abs(report["base_centre"][0] - 0.05) < 0.08
    with pytest.raises(ValueError, match="size must be"):
        place_object(generated_box(), plane=plane, viewmat=VIEW, intrinsic=K, mask=mask_of(target), size="huge")


def test_a_wide_object_is_grown_to_cover_the_old_footprint_within_a_quarter():
    plane = SupportPlane(normal=UP.copy(), offset=0.0, footprint_centre=np.zeros(3), footprint_size=1.0,
                         object_height=0.6, spread=0.001, ring=1000, tilt_deg=0.0, footprint_major=1.0,
                         footprint_minor=0.9)
    # A flat object (a tray): 1.0 wide, 0.4 tall, 0.5 deep in TripoSplat's frame (up along -Y).
    flat = box(6000, (1.0, 0.4, 0.5), seed=11)
    target = box(6000, (0.6, 0.36, 0.24), centre=(0.0, 0.0, 0.12), seed=12)
    _placed, report = place_object(flat, plane=plane, viewmat=VIEW, intrinsic=K, mask=mask_of(target), size="match")
    assert report["footprint_dominant"] is True
    # Matching sqrt(0.9) with sqrt(0.5) gives 1.34; covering the 0.9 width needs 1.8; the boost stops at 1.25 x.
    assert report["cover_scale"] == pytest.approx(1.8, rel=0.05)
    match = report["removed_dominant_extent"] / report["new_dominant_extent_unscaled"]
    assert match == pytest.approx(math.sqrt(0.9 / 0.5), rel=0.05) and report["scale"] == pytest.approx(1.25 * match, rel=1e-6)
    assert footprint_extents(np.random.default_rng(0).random((5000, 2)) * [2.0, 1.0]) == pytest.approx((1.92, 0.96), abs=0.03)


def _table_scene(n_side: int = 160):
    """A peach table (z = 0) over [-1, 1]^2, an object 0.4 x 0.4 x 0.6 on it at the origin, and dark table
    Gaussians under the object at the contact line (the mat in its shadow, which Extract leaves behind)."""
    g = (np.arange(n_side) + 0.5) / n_side * 2 - 1
    table = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    under = (np.abs(table) < 0.2).all(axis=1)
    table = np.c_[table, np.zeros(len(table))]
    rng = np.random.default_rng(13)
    obj = (rng.random((12000, 3)) - 0.5) * [0.4, 0.4, 0.6] + [0, 0, 0.32]
    means = np.vstack([table, obj])
    rgb = np.vstack([np.where(under[:, None], [0.15, 0.1, 0.1], [0.9, 0.6, 0.5]), np.tile([0.9, 0.8, 0.1], (len(obj), 1))])
    ids = np.arange(len(table), len(means))
    return means, rgb, ids, under


def test_inverse_extract_base_drops_the_object_and_the_dark_contact_only(tmp_path):
    means, rgb, ids, under = _table_scene()
    plane = support_plane(means, np.full(len(means), 4.0), ids, UP)
    drop, fill, report = replace_fill(means, np.full(len(means), 4.0), rgb, ids, plane)
    # Every object row goes; of the table only the dark contact under the footprint (grown by 3 %).
    assert drop[ids].all() and report["object_gaussians"] == len(ids)
    table = np.arange(len(means) - len(ids))
    assert drop[table][under].all()
    gone = drop[table] & ~under
    assert np.abs(means[table][gone, :2]).max() < 0.26            # nothing beyond the footprint and its margin
    ring = (np.abs(means[table, :2]) > 0.3).any(axis=1)
    assert not drop[table][ring].any()                            # the ring round it untouched
    # The fill: flat discs on the plane covering the whole old footprint, coloured from the ring (peach).
    assert report["filled_fraction"] == 1.0 and report["source"] == "plane" and fill.count == report["plane_fill_discs"]
    assert np.allclose(plane.height(fill.means), 0.002 * plane.object_height, atol=1e-6)
    assert np.abs(fill.means[:, :2]).max() < 0.27 and np.allclose(fill.rgb, [0.9, 0.6, 0.5], atol=0.02)
    # Written: the kept source rows unchanged and in order, the fill after them.
    _ply(tmp_path / "src.ply", means)
    source = GaussianPly.load(tmp_path / "src.ply")
    count = write_scene(source, ~drop, fill, tmp_path / "base.ply", "base")
    out = GaussianPly.load(tmp_path / "base.ply")
    kept = int((~drop).sum())
    assert count == kept + fill.count and np.array_equal(out.means[:kept], source.means[~drop])
    assert object_rows(source.means, out.means[:kept]).tolist() == np.flatnonzero(~drop).tolist()


def test_fill_covers_only_the_footprint_the_new_object_leaves_bare():
    means, rgb, ids, _under = _table_scene()
    plane = support_plane(means, np.full(len(means), 4.0), ids, UP)
    # A new object on the half x > 0 of the old footprint (dense enough that its lowest tenth, where it
    # touches the plane, closes into a footprint).
    new = box(30000, (0.22, 0.44, 0.5), centre=(0.11, 0.0, 0.25), seed=14)
    _drop, fill, report = replace_fill(means, np.full(len(means), 4.0), rgb, ids, plane, new_object=new)
    assert 0.35 < report["filled_fraction"] < 0.65
    # The bare half only (reaching a cell under the new rim); a few corner cells of the grown old footprint.
    assert (fill.means[:, 0] > 0.03).sum() <= 0.01 * fill.count
    # With a Remove fill: its Gaussians in the bare part, laid flat; the raised sheet and the rest unused.
    rng = np.random.default_rng(15)
    n = 20000
    sheet = np.c_[(rng.random((n, 2)) - 0.5) * 0.6, np.full(n, 0.03)]
    remove = ObjectGaussians(sheet, np.full((n, 3), math.log(0.01)), np.tile([1.0, 0, 0, 0], (n, 1)),
                             np.full(n, 5.0), np.tile([0.7, 0.5, 0.4], (n, 1)))
    _drop, fill, report = replace_fill(means, np.full(len(means), 4.0), rgb, ids, plane, new_object=new,
                                       remove_fill=remove)
    assert report["source"].startswith("remove fill") and report["from_remove_fill"] > 1000
    flat = fill.subset(np.arange(report["from_remove_fill"]))
    assert np.allclose(plane.height(flat.means), 0.002 * plane.object_height, atol=1e-6)
    assert (flat.means[:, 0] > 0.03).sum() <= 0.01 * flat.count and np.allclose(flat.rgb, [0.7, 0.5, 0.4])
    assert report["plane_fill_discs"] < 0.1 * report["from_remove_fill"]


def test_replace_removes_the_old_objects_leftovers_in_its_volume_not_the_ring():
    """Kitchen, 27 Oct 2026: parts of the bulldozer the selection never took (opaque, elongated, inside it)
    showed through the vase as dark and yellow streaks; dark contact rows under the vase's belly as splats."""
    means, rgb, ids, under = _table_scene()
    n_table = len(means) - len(ids)
    logits = np.full(len(means), 4.0)
    rng = np.random.default_rng(21)
    # Missed object rows inside the footprint, above the plane: opaque, yellow and dark.
    missed = np.c_[(rng.random((40, 2)) - 0.5) * 0.36, 0.1 + rng.random(40) * 0.45]
    missed_rgb = np.where(np.arange(40)[:, None] % 2 == 0, [0.9, 0.8, 0.1], [0.1, 0.08, 0.02])
    # A large, faint floater inside the old object's volume.
    floater = np.array([[0.05, -0.05, 0.3]])
    # Dark (object-coloured) rows on the plane under the new object's belly, outside its contact.
    dark_contact = np.c_[(rng.random((30, 2)) - 0.5) * [0.06, 0.3] + [0.17, 0.0], np.full(30, 0.002)]
    # Mat-coloured rows on the plane under the new object's foot (in the old volume, not filled): kept.
    mat_under_foot = np.c_[(rng.random((30, 2)) - 0.5) * 0.06, np.full(30, 0.001)]
    # Beyond the volume: an object-coloured row on the ring, and something above the old object.
    beyond = np.array([[0.5, 0.0, 0.01], [0.0, 0.0, 1.5]])
    extra = np.vstack([missed, floater, dark_contact, mat_under_foot, beyond])
    extra_rgb = np.vstack([missed_rgb, [[0.3, 0.3, 0.3]], np.tile([0.1, 0.08, 0.02], (30, 1)),
                           np.tile([0.9, 0.6, 0.5], (30, 1)), [[0.9, 0.8, 0.1], [0.9, 0.8, 0.1]]])
    first = len(means)
    means = np.vstack([means, extra])
    rgb = np.vstack([rgb, extra_rgb])
    logits = np.r_[logits, np.full(len(extra), 4.0)]
    logits[first + 40] = -2.5                                             # the floater: about 8 % opaque
    rows = {"missed": np.arange(first, first + 40), "floater": np.array([first + 40]),
            "dark_contact": np.arange(first + 41, first + 71), "mat_under_foot": np.arange(first + 71, first + 101),
            "ring": np.array([first + 101]), "above": np.array([first + 102])}
    plane = support_plane(means, logits, ids, UP)
    # A narrow-footed new object: wide (the old footprint) at mid height, touching the plane only round x = 0.
    body = box(30000, (0.4, 0.4, 0.4), centre=(0.0, 0.0, 0.35), seed=22)
    foot = box(6000, (0.12, 0.12, 0.15), centre=(0.0, 0.0, 0.075), seed=23)
    new = ObjectGaussians.concat(foot, body)
    drop, _fill, report = replace_fill(means, logits, rgb, ids, plane, new_object=new)
    assert drop[ids].all()
    assert drop[rows["missed"]].all() and drop[rows["floater"]].all()
    assert drop[rows["dark_contact"]].all()
    assert not drop[rows["mat_under_foot"]].any() and not drop[rows["ring"]].any() and not drop[rows["above"]].any()
    assert report["volume_rows_dropped"] == 41 and report["leftover_contact_rows_dropped"] >= 30
    # The table: under the old footprint the dark contact goes; nothing of it beyond the footprint and margin.
    table = np.arange(n_table)
    assert drop[table][under].all()
    assert np.abs(means[table][drop[table] & ~under, :2]).max() < 0.26
    assert not drop[table][(np.abs(means[table, :2]) > 0.3).any(axis=1)].any()
    # The fill reaches under the new object's belly (it touches the plane only at its foot).
    assert report["filled_fraction"] > 0.8


def test_fill_rows_are_found_by_position_not_row_number():
    rng = np.random.default_rng(6)
    source = rng.random((500, 3)).astype(np.float32)
    births = rng.random((40, 3)).astype(np.float32) + 2
    # The all-around fill writes the kept rows first and its trained (moved) ones after them.
    moved = source[:10] + 0.01
    scene = np.vstack([source[10:300], births, moved])
    flags = fill_rows(scene, source)
    assert flags.sum() == 50 and not flags[:290].any() and flags[290:].all()


def test_contact_shadow_lies_flat_on_the_plane_round_the_footprint():
    obj = box(20000, (0.4, 0.4, 0.8), centre=(0, 0, 0.4), seed=7)
    shadow = contact_shadow(obj, PLANE)
    assert shadow.count > 1000
    height = PLANE.height(shadow.means)
    assert np.allclose(height, 0.01 * np.percentile(PLANE.height(obj.means), 99), atol=1e-9)
    assert (shadow.opacity <= 0.3 + 1e-6).all() and (shadow.rgb == 0).all()
    # Darkest under the object, fading beyond its footprint.
    centre = np.linalg.norm(shadow.means[:, :2], axis=1) < 0.1
    edge = np.linalg.norm(shadow.means[:, :2], axis=1) > 0.26
    assert shadow.opacity[centre].mean() > 0.25 > 0.1 > shadow.opacity[edge].mean()
    # Flat: the third scale is a twentieth of the others.
    assert np.allclose(np.exp(shadow.log_scales[:, 2]), 0.05 * np.exp(shadow.log_scales[:, 0]))
    # Draped over a sheet the cleanup left 0.04 above the plane beyond x = 0.3.
    sheet = np.c_[np.random.default_rng(9).random((20000, 2)) * 2 - 1, np.zeros(20000)]
    sheet[:, 2] = np.where(sheet[:, 0] > 0.3, 0.04, 0.0)
    draped = contact_shadow(obj, PLANE, surface=sheet, surface_band=0.06)
    lift = 0.01 * np.percentile(PLANE.height(obj.means), 99)
    far = draped.means[:, 0] > 0.33
    assert np.allclose(draped.means[far, 2], 0.04 + lift) and np.allclose(draped.means[draped.means[:, 0] < 0.25, 2], lift)


def _ply(path, means, opacity=5.0, extra_rest: bool = True):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"] + ([f"f_rest_{i}" for i in range(9)] if extra_rest else []) + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(len(means), dtype=[(n, "<f4") for n in names])
    for i, c in enumerate("xyz"):
        v[c] = means[:, i]
    v["opacity"] = opacity
    for i in range(3):
        v[f"scale_{i}"] = math.log(0.01)
    v["rot_0"] = 1
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def test_largest_component_load_and_scene_write(tmp_path):
    rng = np.random.default_rng(8)
    body = rng.random((3000, 3)) * 0.5
    card = rng.random((200, 3)) * [0.3, 0.3, 0.01] + [3, 3, 3]      # a background card far away
    _ply(tmp_path / "gen.ply", np.vstack([body, card]), extra_rest=False)
    obj = load_object_ply(tmp_path / "gen.ply")
    kept, report = keep_largest_component(obj)
    assert kept.count == 3000 and report["components"] >= 2
    assert np.allclose(kept.rgb, 0.5)
    _ply(tmp_path / "scene.ply", rng.random((100, 3)))
    scene = GaussianPly.load(tmp_path / "scene.ply")
    keep = np.ones(100, bool)
    keep[:10] = False
    moved = (np.array([20, 21]), np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]))
    count = write_scene(scene, keep, kept, tmp_path / "out.ply", "test", moved=moved)
    out = GaussianPly.load(tmp_path / "out.ply")
    assert count == len(out.vertex) == 90 + 3000 and out.max_sh_degree == 1
    assert np.allclose(out.means[90:], kept.means, atol=1e-5)
    assert np.allclose(out.rest[90:], 0)
    assert np.allclose(out.means[10:12], [[5, 5, 5], [6, 6, 6]]) and np.allclose(out.means[12], scene.means[22])


def _node_inputs(tmp_path):
    """The table scene as Graphdeco PLYs for the Replace nodes: the source, its extract, a Remove result (the
    table without the object, its footprint filled with green discs) and two level cameras round it."""
    from types import SimpleNamespace

    means, _rgb, ids, under = _table_scene()
    _ply(tmp_path / "source.ply", means)
    _ply(tmp_path / "object_only.ply", means[ids])
    table = means[: len(means) - len(ids)]
    g = np.linspace(-0.2, 0.2, 40)
    discs = np.c_[np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2), np.full(len(g) ** 2, 0.001)]
    _ply(tmp_path / "removed.ply", np.vstack([table[~under], discs]))
    views = [SimpleNamespace(view_id=f"camera_{i}", width=256, height=256, viewmat=v, intrinsic=K, is_virtual=False)
             for i, v in enumerate((VIEW, look_at([3.0, 0.0, 2.0], [0.0, 0.0, 0.3])))]
    cameras = SimpleNamespace(cameras=SimpleNamespace(by_id={v.view_id: v for v in views}, views=views))
    return SimpleNamespace(ply=tmp_path / "source.ply"), cameras, len(ids)


def _nodes():
    try:                                    # this repository
        from competition.nodes import NODE_CLASS_MAPPINGS
    except ImportError:                     # the exported node pack has it at its root
        from nodes import NODE_CLASS_MAPPINGS
    return NODE_CLASS_MAPPINGS


def test_replace_base_node_builds_on_the_remove_result_as_it_is(tmp_path):
    from PIL import Image

    scene, cameras, n_object = _node_inputs(tmp_path)
    Image.new("RGB", (256, 256)).save(tmp_path / "hero.png")
    node = _nodes()["SpEdE_LocalReplaceBase"]()
    args = dict(scene=scene, cameras=cameras, view_id="camera_0", object_ply=str(tmp_path / "object_only.ply"),
                device="cpu", hero_image_path=str(tmp_path / "hero.png"))
    # Without a Remove: the source without the object rows, the footprint filled.
    base, _hero, report = node.base(output_path=str(tmp_path / "src_base.ply"), **args)
    report = json.loads(report)
    assert report["base"] == "source" and report["object_gaussians"] == n_object and report["filled_fraction"] > 0
    # On a Remove: its scene unchanged, row for row (the remove fill is ignored there); the plane still measured.
    removed = GaussianPly.load(tmp_path / "removed.ply")
    base, _hero, report = node.base(output_path=str(tmp_path / "rm_base.ply"), remove_scene_ply=str(tmp_path / "removed.ply"),
                                    remove_fill_ply=str(tmp_path / "removed.ply"), **args)
    report = json.loads(report)
    out = GaussianPly.load(base.ply)
    assert report["base"] == "remove" and report["total_gaussians"] == len(removed.vertex) == base.count
    assert np.array_equal(out.vertex, removed.vertex) and "filled_fraction" not in report
    assert report["plane"]["object_height"] == pytest.approx(0.6, abs=0.05)


def test_place_generated_node_on_the_remove_result_adds_only_the_object_and_its_shadow(tmp_path):
    scene, cameras, n_object = _node_inputs(tmp_path)
    _ply(tmp_path / "generated.ply", generated_box().means, extra_rest=False)
    generated = type("Splat", (), {"get_bytes": lambda self: (tmp_path / "generated.ply").read_bytes()})()
    target = box(6000, (0.4, 0.4, 0.6), centre=(0.0, 0.0, 0.3), seed=2)
    picture, alpha = point_render(target, VIEW, K, 256, 256)
    node = _nodes()["SpEdE_LocalPlaceGenerated"]()
    args = dict(scene=scene, cameras=cameras, view_id="camera_0", object_ply=str(tmp_path / "object_only.ply"),
                generated=generated, inserted_image=picture[None], object_mask=alpha[None], max_shift=0.25,
                size="match", device="cpu")
    _out, report = node.place(output_path=str(tmp_path / "a/replaced_scene.ply"), **args)
    report = json.loads(report)
    assert report["base"] == "source" and report["remove_scene_ply"] is None and report["fill"]["filled_fraction"] >= 0
    # On the Remove's result: its rows first, unchanged and complete; then only the new object and its shadow.
    removed = GaussianPly.load(tmp_path / "removed.ply")
    out, report = node.place(output_path=str(tmp_path / "b/replaced_scene.ply"),
                             remove_scene_ply=str(tmp_path / "removed.ply"), **args)
    report = json.loads(report)
    rows = GaussianPly.load(out.ply).vertex
    n = len(removed.vertex)
    assert report["base"] == "remove" and report["remove_scene_ply"] == str(tmp_path / "removed.ply")
    assert report["fill"] is None and report["fill_gaussians"] == 0 and report["remove_fill_ply"] is None
    assert report["total_gaussians"] == n + report["object_gaussians"] + report["shadow_gaussians"] == len(rows)
    assert np.array_equal(rows[:n], removed.vertex)
    # The size still comes from the extract: match makes the new object about as big as the old one.
    assert report["placement"]["height_vs_removed"] == pytest.approx(1.0, abs=0.25)
