import warnings
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from spede_core.ply import GaussianPly
from spede_core.surface_fill import fit_height_field, support_surface_fill, surface_height


def _write(path: Path, xyz: np.ndarray, dc: np.ndarray, opacity_logit: float = 4.0, rest: np.ndarray | None = None):
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"] + [f"f_rest_{i}" for i in range(9)]
    v = np.zeros(len(xyz), dtype=[(n, "<f4") for n in names])
    for i, k in enumerate("xyz"):
        v[k] = xyz[:, i]
    for i in range(3):
        v[f"f_dc_{i}"] = dc[:, i]
    v["opacity"] = opacity_logit
    v["scale_0"] = v["scale_1"] = v["scale_2"] = np.log(0.01)
    v["rot_0"] = 1.0
    if rest is not None:                                  # (N, 3 coefficients, 3 colours)
        for c in range(3):
            for k in range(3):
                v[f"f_rest_{c * 3 + k}"] = rest[:, k, c]
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)


def test_height_field_ignores_outliers():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-1, 1, 400), rng.uniform(-1, 1, 400)
    h = 0.1 * x - 0.05 * y + 0.2 * x * x
    h[:40] += 0.5                                   # 10 % clutter above the surface
    coef, scale = fit_height_field(x, y, h)
    assert np.abs(surface_height(coef, x[40:], y[40:]) - h[40:]).max() < 1e-3
    assert scale < 0.01


def test_support_surface_replaces_residue_with_a_flat_layer(tmp_path):
    rng = np.random.default_rng(1)
    # A tilted table (up = z) around an object standing at the origin; under
    # the object, dark residue above the table; far away, a wall to keep.
    tx, ty = rng.uniform(-1, 1, 6000), rng.uniform(-1, 1, 6000)
    table = np.c_[tx, ty, 0.05 * tx]
    residue = np.c_[rng.uniform(-0.13, 0.13, 300), rng.uniform(-0.13, 0.13, 300), rng.uniform(0.0, 0.1, 300)]
    residue[:, 2] += 0.05 * residue[:, 0]
    wall = np.c_[rng.uniform(-1, 1, 500), np.full(500, 3.0), rng.uniform(0, 1, 500)]
    table = table[(np.abs(table[:, 0]) > 0.2) | (np.abs(table[:, 1]) > 0.2)]   # unseen under the object
    xyz = np.r_[table, residue, wall]
    dc = np.r_[np.full((len(table), 3), 0.8), np.full((len(residue), 3), -1.5), np.zeros((len(wall), 3))]
    scene = tmp_path / "removed.ply"
    rest = np.zeros((len(xyz), 3, 3))
    rest[:len(table), 0, :] = (-0.3, -0.2, -0.1)                 # the table darkens towards one side
    _write(scene, xyz, dc, rest=rest)
    obj = np.c_[rng.uniform(-0.15, 0.15, 2000), rng.uniform(-0.15, 0.15, 2000), rng.uniform(0.0, 0.8, 2000)]
    axes = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    out = tmp_path / "filled.ply"
    rep = support_surface_fill(scene_ply=scene, object_means=obj, axes=axes, out_ply=out, log=lambda _t: None)
    g = GaussianPly.load(out)
    assert rep["carved"] >= 290                                   # the residue is gone
    assert not (g.dc[:, 0, 0] < -1).any()
    assert np.isclose(g.means[:, 1], 3.0).sum() == 500           # the wall is kept
    discs = g.means[len(g.means) - rep["discs"]:]
    assert rep["discs"] > 100
    assert np.abs(discs[:, 2] - 0.05 * discs[:, 0]).max() < 0.01  # on the table's surface
    assert np.allclose(g.rest[len(g.means) - rep["discs"]:, 0], (-0.3, -0.2, -0.1))   # the table's view terms
    # discs cover the whole footprint
    assert np.abs(discs[:, :2]).max() > 0.15 and (np.abs(discs[:, :2]) < 0.05).all(1).any()


def test_discs_take_their_colour_from_a_clean_view(tmp_path):
    from spede_core.surface_fill import SH_C0, colour_discs_from_view
    xyz = np.array([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [9.0, 0.0, 2.0]])      # last one outside the image
    path = tmp_path / "d.ply"
    _write(path, xyz, np.zeros((3, 3)))
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, :32] = (200, 100, 50)
    image[:, 32:] = (20, 40, 60)
    K = np.array([[32.0, 0, 32], [0, 32.0, 32], [0, 0, 1]])
    assert colour_discs_from_view(path, 2, image, np.eye(4), K) == 1
    g = GaussianPly.load(path)
    rgb = g.dc[:, 0] * SH_C0 + 0.5
    assert np.allclose(rgb[0], 0.5)                                   # not a disc: untouched
    assert np.allclose(rgb[1], np.array([20, 40, 60]) / 255, atol=1e-3)
    assert np.allclose(rgb[2], rgb[1], atol=1e-3)                     # outside: the median


def test_harmonised_discs_follow_the_surrounding_gradient(tmp_path):
    from spede_core.surface_fill import SH_C0, harmonise_disc_colours
    rng = np.random.default_rng(3)
    # Real table: brightness rising along x; discs in the middle all at 0.6.
    tx, ty = rng.uniform(-1, 1, 20000), rng.uniform(-1, 1, 20000)
    keep = (np.abs(tx) > 0.4) | (np.abs(ty) > 0.4)
    tx, ty = tx[keep], ty[keep]
    table_rgb = np.clip(0.4 + 0.2 * tx, 0, 1)
    gx, gy = np.meshgrid(np.arange(-0.4, 0.4, 0.01), np.arange(-0.4, 0.4, 0.01))
    gx, gy = gx.ravel(), gy.ravel()
    xyz = np.r_[np.c_[tx, ty, np.zeros_like(tx)], np.c_[gx, gy, np.zeros_like(gx)]]
    rgb = np.r_[np.repeat(table_rgb[:, None], 3, 1), np.full((len(gx), 3), 0.6)]
    path = tmp_path / "s.ply"
    _write(path, xyz, (rgb - 0.5) / SH_C0)
    axes = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    rep = harmonise_disc_colours(path, np.c_[gx, gy, np.zeros_like(gx)], axes)
    assert rep["harmonised"] == len(gx)
    g = GaussianPly.load(path)
    out = (g.dc[len(tx):, 0, 0] * SH_C0 + 0.5)
    left, right = out[gx < -0.3].mean(), out[gx > 0.3].mean()
    assert abs(left - (0.4 + 0.2 * -0.35)) < 0.05 and abs(right - (0.4 + 0.2 * 0.35)) < 0.05


def test_disc_layer_feathers_over_the_surrounding_surface(tmp_path):
    rng = np.random.default_rng(5)
    tx, ty = rng.uniform(-1, 1, 20000), rng.uniform(-1, 1, 20000)
    keep = (np.abs(tx) > 0.2) | (np.abs(ty) > 0.2)
    xyz = np.c_[tx[keep], ty[keep], np.zeros(keep.sum())]
    scene = tmp_path / "removed.ply"
    _write(scene, xyz, np.full((len(xyz), 3), 0.5))
    obj = np.c_[rng.uniform(-0.15, 0.15, 3000), rng.uniform(-0.15, 0.15, 3000), rng.uniform(0.0, 0.8, 3000)]
    axes = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
    out = tmp_path / "filled.ply"
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("error", RuntimeWarning)       # the Huber fit stays warning-free
        rep = support_surface_fill(scene_ply=scene, object_means=obj, axes=axes, out_ply=out, feather=0.1,
                                   log=lambda _t: None)
    g = GaussianPly.load(out)
    discs = slice(len(g.means) - rep["discs"], None)
    a = 1 / (1 + np.exp(-g.opacity_logits[discs, 0]))
    r = np.abs(g.means[discs, :2]).max(1)
    assert rep["feathered"] > 0
    assert a[r < 0.1].min() > 0.9                       # inside: opaque
    assert a[r > r.max() - 0.01].max() < 0.5            # outer edge: faded
    assert len(g.means) - rep["discs"] == len(xyz) - rep["carved"]   # nothing outside the fill carved


def _down_camera(height: float, f: float = 100.0, size: int = 64):
    """A camera at (0, 0, height) looking straight down; viewmat, K."""
    vm = np.eye(4)
    vm[:3, :3] = np.diag([1.0, -1.0, -1.0])
    vm[:3, 3] = (0, 0, height)
    K = np.array([[f, 0, size / 2], [0, f, size / 2], [0, 0, 1.0]])
    return vm, K


def test_discs_textured_from_the_view_that_sees_them_best(tmp_path):
    from plyfile import PlyData

    from spede_core.surface_fill import SH_C0, texture_discs_from_views

    gx, gy = np.meshgrid(np.linspace(-0.1, 0.1, 9), np.linspace(-0.1, 0.1, 9))
    xyz = np.c_[gx.ravel(), gy.ravel(), np.zeros(81)]
    raw, path = tmp_path / "raw.ply", tmp_path / "discs.ply"
    _write(raw, xyz, np.zeros((81, 3)))
    v = PlyData.read(raw)["vertex"].data.copy()
    v["scale_2"] = np.log(1e-4)                                  # flat discs, normal +z
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)
    near_vm, near_K = _down_camera(1.0)
    far_vm, far_K = _down_camera(3.0)
    red = np.zeros((64, 64, 3), np.uint8); red[..., 0] = 255
    blue = np.zeros((64, 64, 3), np.uint8); blue[..., 2] = 255
    region = np.ones((64, 64), bool)

    def views(near_depth):
        return [{"name": "near", "image": red, "viewmat": near_vm, "intrinsic": near_K,
                 "depth": np.full((64, 64), near_depth), "region": region},
                {"name": "far", "image": blue, "viewmat": far_vm, "intrinsic": far_K,
                 "depth": np.full((64, 64), 3.0), "region": region}]

    rep = texture_discs_from_views(path, xyz, views(1.0))
    rgb = GaussianPly.load(path).dc[:, 0] * SH_C0 + 0.5
    assert rep["textured"] == 81 and rep["lead_view_counts"] == {"near": 81}
    assert rgb[:, 0].min() > 0.95 and rgb[:, 2].max() < 0.05
    # Something in front of the discs in the near view: the far view's colour.
    texture_discs_from_views(path, xyz, views(0.5))
    rgb = GaussianPly.load(path).dc[:, 0] * SH_C0 + 0.5
    assert rgb[:, 2].min() > 0.95 and rgb[:, 0].max() < 0.05


def test_feather_band_width_wanders_along_the_outline(tmp_path):
    rng = np.random.default_rng(6)
    tx, ty = rng.uniform(-1, 1, 20000), rng.uniform(-1, 1, 20000)
    keep = (np.abs(tx) > 0.2) | (np.abs(ty) > 0.2)
    xyz = np.c_[tx[keep], ty[keep], np.zeros(keep.sum())]
    scene = tmp_path / "removed.ply"
    _write(scene, xyz, np.full((len(xyz), 3), 0.5))
    obj = np.c_[rng.uniform(-0.15, 0.15, 3000), rng.uniform(-0.15, 0.15, 3000), rng.uniform(0.0, 0.8, 3000)]
    axes = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))

    def outer_extent(jitter):
        out = tmp_path / f"filled_{jitter}.ply"
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", RuntimeWarning)   # the Huber fit stays warning-free
            rep = support_surface_fill(scene_ply=scene, object_means=obj, axes=axes, out_ply=out, feather=0.1,
                                       feather_jitter=jitter, log=lambda _t: None)
        g = GaussianPly.load(out)
        m = g.means[len(g.means) - rep["discs"]:]
        a = 1 / (1 + np.exp(-g.opacity_logits[len(g.means) - rep["discs"]:, 0]))
        assert a.min() >= 0 and a.max() <= 0.951
        # The band's reach outwards along each side's middle stretch.
        reach = []
        for k in np.linspace(-0.12, 0.12, 13):
            for side in (m[np.abs(m[:, 1] - k) < 0.01, 0], m[np.abs(m[:, 0] - k) < 0.01, 1]):
                reach += [side.max(), -side.min()]
        return np.ptp(reach)

    assert outer_extent(0.6) > outer_extent(0.0) + 0.02


def test_feather_band_fades_to_the_surface_colour(tmp_path):
    from plyfile import PlyData

    from spede_core.surface_fill import SH_C0, crossfade_feather_to_surface

    gx, gy = np.meshgrid(np.linspace(-0.1, 0.1, 9), np.linspace(-0.1, 0.1, 9))
    xyz = np.c_[gx.ravel(), gy.ravel(), np.zeros(81)]
    raw, path = tmp_path / "raw.ply", tmp_path / "discs.ply"
    _write(raw, xyz, np.full((81, 3), (np.array([1.0, 0.0, 0.0]) - 0.5) / SH_C0))   # red discs
    v = PlyData.read(raw)["vertex"].data.copy()
    v["scale_2"] = np.log(1e-4)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)
    vm, K = _down_camera(1.0)
    blue = np.zeros((64, 64, 3), np.uint8); blue[..., 2] = 255
    ramp = np.r_[np.ones(27), np.full(27, 0.5), np.zeros(27)]
    rep = crossfade_feather_to_surface(path, xyz, ramp, {"image": blue, "viewmat": vm, "intrinsic": K,
                                                         "depth": np.full((64, 64), 1.0),
                                                         "region": np.ones((64, 64), bool)})
    rgb = GaussianPly.load(path).dc[:, 0] * SH_C0 + 0.5
    assert rep["crossfaded"] == 54
    assert np.allclose(rgb[:27], [1, 0, 0], atol=1e-4)               # inside: untouched
    assert np.allclose(rgb[27:54], [0.5, 0, 0.5], atol=0.02)         # half way
    assert np.allclose(rgb[54:], [0, 0, 1], atol=0.02)               # outer edge: the surface


def test_restore_outside_keeps_the_surface_layer_past_the_discs(tmp_path):
    rng = np.random.default_rng(5)
    # A table (up = z) seen everywhere but under the object's stem; the
    # object's canopy reaches past the stem, so the carve covers ground the
    # discs do not. There the table carries a light upper layer just above
    # it (kitchen: the mat's top splats over a darker layer), and a twig of
    # the object higher up.
    tx, ty = rng.uniform(-1, 1, 40000), rng.uniform(-1, 1, 40000)
    table = np.c_[tx, ty, np.zeros_like(tx)]
    table = table[(np.abs(table[:, 0]) > 0.17) | (np.abs(table[:, 1]) > 0.17)]
    layer = np.c_[rng.uniform(0.21, 0.24, 200), rng.uniform(-0.1, 0.1, 200), np.full(200, 0.04)]
    twig = np.c_[rng.uniform(0.21, 0.24, 50), rng.uniform(-0.1, 0.1, 50), np.full(50, 0.3)]
    under = np.c_[rng.uniform(-0.1, 0.1, 100), rng.uniform(-0.1, 0.1, 100), np.full(100, 0.04)]
    xyz = np.r_[table, layer, twig, under]
    scene = tmp_path / "removed.ply"
    _write(scene, xyz, np.zeros((len(xyz), 3)))
    stem = np.c_[rng.uniform(-0.15, 0.15, 1500), rng.uniform(-0.15, 0.15, 1500), rng.uniform(0.0, 0.3, 1500)]
    canopy = np.c_[rng.uniform(-0.26, 0.26, 20000), rng.uniform(-0.26, 0.26, 20000), rng.uniform(0.4, 0.8, 20000)]
    obj = np.r_[stem, canopy]
    axes = (np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))

    def kept(**kw):
        out = tmp_path / f"filled_{len(kw)}.ply"
        rep = support_surface_fill(scene_ply=scene, object_means=obj, axes=axes, out_ply=out, log=lambda _t: None,
                                   **kw)
        m = GaussianPly.load(out).means[:-rep["discs"]]
        rows = {r.tobytes() for r in np.ascontiguousarray(m, np.float32)}
        return rep, [sum(r.tobytes() in rows for r in np.ascontiguousarray(p, np.float32))
                     for p in (layer, twig, under)]

    plain, (layer_kept, twig_kept, under_kept) = kept()
    assert (layer_kept, twig_kept, under_kept) == (0, 0, 0)          # the default carves all three
    assert "restored_outside" not in plain
    rep, (layer_kept, twig_kept, under_kept) = kept(restore_outside=True)
    assert layer_kept == 200 and rep["restored_outside"] == 200      # past the discs, near the surface: kept
    assert twig_kept == 0 and under_kept == 0                        # too high / under the discs: carved
    assert rep["carved"] == plain["carved"] - 200
    assert rep["discs"] == plain["discs"]
