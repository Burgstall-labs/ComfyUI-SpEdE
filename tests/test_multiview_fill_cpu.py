"""CPU checks of the all-around fill's geometry (spede_core.multiview_fill)."""
import numpy as np

from spede_core.multiview_fill import (HeroWarp, choose_chain, choose_orbit_stops, frame_check, hero_depth,
                                       held_out_chain, object_hole, orbit_path)
from spede_core.renderer import CameraSet, CameraView

K = np.array([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]], np.float32)


def look_at(eye, target=(0, 0, 0), up=(0, 0, 1)):
    """World-to-camera, OpenCV axes (x right, y down, z forward)."""
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, y, z], 1)
    c2w[:3, 3] = eye
    return np.linalg.inv(c2w).astype(np.float32)


def ring(n=24, radius=4.0, height=1.5):
    views = []
    for i in range(n):
        a = 2 * np.pi * i / n
        views.append(CameraView(view_id=f"c{i:02d}", viewmat=look_at((radius * np.cos(a), radius * np.sin(a), height)),
                                intrinsic=K, width=256, height=256))
    return CameraSet(views)


def rows_for(cams, skip=()):
    return [{"view_id": v.view_id, "score": 1.0 - 0.01 * i, "visible": 0.0 if v.view_id in skip else 0.95,
             "in_frame": 1.0, "area_fraction": 0.1} for i, v in enumerate(cams.views)]


def test_orbit_stops_go_round_and_end_at_the_hero():
    cams = ring()
    stops = choose_orbit_stops(rows_for(cams), cams, "c00", np.zeros(3), spacing=45.0)
    assert stops[-1] == "c00"
    assert 6 <= len(stops) - 1 <= 9                  # about 360 / 45 stops between
    assert len(set(stops[:-1])) == len(stops) - 1
    # A gap without usable cameras is bridged by the next camera after it.
    gap = {f"c{i:02d}" for i in range(4, 11)}
    stops = choose_orbit_stops(rows_for(cams, skip=gap), cams, "c00", np.zeros(3), spacing=45.0)
    assert not gap & set(stops) and stops[-1] == "c00"


def test_orbit_path_stays_on_the_ring_and_looks_at_real_poses_at_stops():
    cams = ring()
    stops = ["c06", "c12", "c18", "c00"]
    path = orbit_path(cams, "c00", stops, np.zeros(3), 124)
    vms = np.asarray(path["viewmats"])
    centres = np.stack([np.linalg.inv(m)[:3, 3] for m in vms])
    radius = np.linalg.norm(centres[:, :2], axis=1)
    assert np.allclose(radius, 4.0, atol=1e-6) and np.allclose(centres[:, 2], 1.5, atol=1e-6)
    for key, v in zip(path["keys"], ["c00", *stops]):
        if float(key).is_integer():
            assert np.allclose(vms[int(key)], cams.by_id[v].viewmat, atol=1e-4)
    assert np.allclose(vms[0], vms[-1], atol=1e-4)          # closed loop


def test_hero_warp_reproduces_the_hero_and_marks_unseen_pixels():
    cams = ring()
    hero = cams.by_id["c00"]
    depth = np.full((256, 256), np.linalg.norm([4.0, 1.5]), np.float64)
    rgb = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
    warp = HeroWarp(rgb, depth, K, hero.viewmat)
    same = warp.render(hero.viewmat, K)
    assert (same == rgb).all(axis=2).mean() > 0.95
    side = warp.render(cams.by_id["c06"].viewmat, K)
    magenta = (side == [255, 0, 255]).all(axis=2).mean()
    assert magenta > 0.3                              # 90 degrees round: mostly unseen


def test_hero_depth_keeps_solid_scene_and_upsamples():
    mask = np.zeros((8, 8), bool)
    mask[2:5, 2:5] = True
    source = np.full((8, 8), 2.0)
    alpha = np.ones((8, 8))
    aligned = np.full((8, 8), 3.0)
    z = hero_depth(mask, source, alpha, aligned)
    assert z[3, 3] == 3.0 and z[0, 0] == 2.0
    assert hero_depth(mask, source, alpha, aligned, scale=2).shape == (16, 16)


def test_frame_check_accepts_the_warp_and_rejects_a_shift():
    rng = np.random.default_rng(1)
    warp = (rng.random((128, 128, 3)) * 255).astype(np.uint8)
    from scipy import ndimage
    warp = ndimage.gaussian_filter(warp, (3, 3, 0))
    assert frame_check(warp, warp)["usable"]
    moved = np.roll(warp, 6, axis=1)
    assert not frame_check(warp, moved)["usable"]


def test_chain_spreads_directions_and_hold_out_skips_the_hero():
    cams = ring()
    chain = choose_chain(rows_for(cams), cams, "c00", np.zeros(3), count=6)["chain"]
    assert chain[0] == "c00" and len(chain) == 6
    held = held_out_chain(chain)
    assert "c00" not in held and held <= set(chain)


def test_object_hole_covers_the_projection():
    cams = ring()
    pts = np.random.default_rng(2).normal(0, 0.2, (500, 3))
    hole = object_hole(pts, cams.by_id["c00"].viewmat, K, 256, dilate=4)
    assert hole[128, 128] and not hole[0, 0]


def test_progressive_visit_order_grows_from_the_hero():
    from spede_core.progressive_fill import visit_order

    class Cam:
        def __init__(self, view_id, angle):
            a = np.radians(angle)
            c2w = np.eye(4)
            c2w[:3, 3] = [3 * np.cos(a), 3 * np.sin(a), 0]
            self.view_id, self.viewmat = view_id, np.linalg.inv(c2w)

    class Cams:
        def __init__(self, cams):
            self.by_id = {c.view_id: c for c in cams}

    angles = {"h": 0, "a": 20, "b": -22, "c": 45, "d": 180, "e": 90}
    cams = Cams([Cam(k, v) for k, v in angles.items()])
    order = visit_order(cams, list(angles), "h", np.zeros(3))
    assert [v for v, _ in order] == ["a", "b", "c", "e", "d"]
    assert dict(order) == {"a": "h", "b": "h", "c": "a", "e": "c", "d": "e"}


def test_seam_tone_matching_brightens_a_darkened_fill():
    from spede_core.progressive_fill import match_seam_tone
    render = np.full((80, 80, 3), 120, np.uint8)
    region = np.zeros((80, 80), bool)
    region[20:60, 20:60] = True
    edited = render.copy()
    edited[region] = 96                                  # 20 % darker inside
    out, gain = match_seam_tone(edited, render, region, band=6)
    assert np.allclose(gain, 1.25)
    assert abs(float(out[region].mean()) - 120) < 1
    assert (out[~region] == 120).all()


def test_seam_tone_field_follows_a_gradient():
    from spede_core.progressive_fill import match_seam_tone_field
    x = np.linspace(80, 160, 256)
    render = np.repeat(np.repeat(x[None, :, None], 256, 0), 3, 2).astype(np.uint8)
    region = np.zeros((256, 256), bool)
    region[64:192, 64:192] = True
    edited = render.copy()
    edited[region] = 100
    out, _gain = match_seam_tone_field(edited, render, region, band=12, work=128)
    left, right = out[128, 70, 0], out[128, 186, 0]
    assert abs(int(left) - int(render[128, 60, 0])) < 12
    assert abs(int(right) - int(render[128, 196, 0])) < 12
    assert (out[~region] == render[~region]).all()


def test_fill_tone_target_matches_the_surroundings():
    from spede_core.progressive_fill import fill_tone_target

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:256, :256]
    rgb = 100 + rng.normal(0, 5, (256, 256, 3))
    fill = (yy - 128) ** 2 + (xx - 128) ** 2 < 50 ** 2
    rgb[fill] = 160 + rng.normal(0, 30, (fill.sum(), 3))     # a lighter, high-contrast patch
    out = fill_tone_target(rgb, fill, band=24, work=256)
    assert abs(out[fill].mean() - 100) < 4
    assert out[fill].std() < 0.4 * rgb[fill].std()
    assert np.array_equal(out[~fill], rgb[~fill])


def test_top_camera_looks_straight_down_on_the_discs():
    from types import SimpleNamespace

    from spede_core.geometry_utils import project_to_pixels
    from spede_core.progressive_fill import top_camera

    rng = np.random.default_rng(4)
    discs = np.c_[rng.uniform(1, 2, 500), rng.uniform(-1, 0, 500), np.zeros(500)]
    vm = np.eye(4); vm[:3, 3] = (0, 0, 3.0)                      # a real camera 3 units away
    cams = SimpleNamespace(views=[SimpleNamespace(viewmat=vm, is_virtual=False)])
    up = np.array([0, 0, 1.0])
    cam = top_camera(discs, (up, np.array([1.0, 0, 0]), np.array([0, 1.0, 0])), cams, 512)
    uv, z = project_to_pixels(discs, cam.viewmat, cam.intrinsic)
    assert (z > 0).all() and np.ptp(z) < 1e-6                     # all at one depth: straight down
    assert (uv > 0).all() and (uv < 512).all()
    assert np.ptp(uv[:, 0]) > 200                                 # and they fill a good part of it
    centre = np.linalg.inv(cam.viewmat)[:3, 3]
    assert centre[2] > 2


def test_magenta_left_finds_unfilled_marker():
    from spede_core.progressive_fill import _magenta_left

    img = np.full((8, 8, 3), 120, np.uint8)
    img[2:4, 2:4] = (250, 30, 240)
    assert _magenta_left(img).sum() == 4


def test_magenta_tinted_flags_a_purple_wash():
    from spede_core.progressive_fill import _magenta_tinted

    img = np.full((10, 10, 3), 120, np.uint8)
    region = np.zeros((10, 10), bool); region[:5] = True
    assert _magenta_tinted(img, region) == 0.0
    img[:5] = (170, 90, 160)                                      # the wash an edit painted
    assert _magenta_tinted(img, region) == 1.0


def test_real_surface_around_the_fill_fades_to_the_painted_top_view(tmp_path):
    from plyfile import PlyData, PlyElement

    from spede_core.ply import GaussianPly
    from spede_core.progressive_fill import recolour_surface_under
    from spede_core.surface_fill import SH_C0

    # A row of flat Gaussians on z = 0 (x from -0.3 to 0.3), plus one 0.2
    # above the surface; the disc is the one at the origin.
    xs = np.linspace(-0.3, 0.3, 7)
    xyz = np.r_[np.c_[xs, np.zeros(7), np.zeros(7)], [[0.2, 0.0, 0.2]]].astype(np.float32)
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(len(xyz), dtype=[(n, "<f4") for n in names])
    for i, k in enumerate("xyz"):
        v[k] = xyz[:, i]
    for c, val in enumerate((1.0, 0.0, 0.0)):                       # all red
        v[f"f_dc_{c}"] = (val - 0.5) / SH_C0
    v["opacity"] = 4.0
    v["scale_0"] = v["scale_1"] = np.log(0.01)
    v["scale_2"] = np.log(1e-4)
    v["rot_0"] = 1.0
    path = tmp_path / "scene.ply"
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(path)
    vm = np.eye(4)
    vm[:3, :3] = np.diag([1.0, -1.0, -1.0])
    vm[:3, 3] = (0, 0, 1.0)                                       # 1 above, looking down
    K = np.array([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]])
    blue = np.zeros((128, 128, 3), np.uint8)
    blue[..., 2] = 255
    weight = np.clip(1 - np.abs(np.arange(128) - 64) / 25.0, 0, 1)[None].repeat(128, 0)   # 1 at x = 0, 0 by |x| = 0.25
    n = recolour_surface_under(path, xyz[3:4], blue, weight, vm, K, np.ones((128, 128)))
    rgb = GaussianPly.load(path).dc[:, 0] * SH_C0 + 0.5
    assert n == 4                                                 # x = +-0.1, +-0.2 (the disc and the floater not)
    assert np.allclose(rgb[3], [1, 0, 0], atol=1e-4)              # the disc itself untouched
    assert np.allclose(rgb[7], [1, 0, 0], atol=1e-4)              # above the surface: untouched
    assert np.allclose(rgb[[0, 6]], [1, 0, 0], atol=1e-4)         # past the ring
    assert rgb[2, 2] > 0.5 and rgb[1, 2] < rgb[2, 2]              # fading outwards


def test_independent_fill_cleans_each_camera_on_its_own_photo(tmp_path, monkeypatch):
    """The per-camera part of independent_fill on CPU (``refit=False``, the
    23 October fill): the support surface, the optimisation and the
    finishing (gsplat) are stubbed."""
    from pathlib import Path

    from plyfile import PlyData, PlyElement

    import spede_core.progressive_fill as pf

    cams = ring(n=8)
    rng = np.random.default_rng(0)
    obj = rng.uniform(-0.3, 0.3, (400, 3))
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(len(obj), dtype=[(n, "<f4") for n in names])
    for i, k in enumerate("xyz"):
        v[k] = obj[:, i]
    v["rot_0"] = 1.0
    source = tmp_path / "source.ply"
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(source)
    photos = {c.view_id: np.full((256, 256, 3), 120, np.uint8) for c in cams.views}
    seeds = []

    def clean_single(image, seed):
        assert image.shape == (128, 128, 3)
        seeds.append(seed)
        out = image.copy()
        out[40:90, 40:90] = 90                  # the object "removed"
        return out

    calls = {}
    filled = tmp_path / "out" / "support_surface.ply"

    def support(**kw):
        calls["support"] = kw
        filled.write_bytes(b"ply")
        return {"ply": filled, "surface": {"discs": 3}, "disc_means": np.zeros((3, 3)), "disc_ramp": np.ones(3),
                "protect": np.zeros((0, 3)), "top": None, "top_tiles": [], "Kh": None}

    def optimise(**kw):
        calls["optimise"] = kw
        Path(kw["out_ply"]).write_bytes(b"ply")
        return {"after": {}}

    monkeypatch.setattr(pf, "_support_surface", support)
    monkeypatch.setattr(pf, "multiview_optimize", optimise)
    monkeypatch.setattr(pf, "_finish_surface", lambda final, **kw: calls.setdefault("finish", kw))
    progress = []
    rep = pf.independent_fill(source_ply=source, start_ply=tmp_path / "depth_patch_scene.ply", cameras=cams,
                              hero="c00", hero_clean=np.full((64, 64, 3), 100, np.uint8), photos=photos,
                              selections={}, object_ids=np.arange(len(obj)), centre=np.zeros(3),
                              clean=lambda *_a: None, clean_single=clean_single, out_dir=tmp_path / "out",
                              size=128, seed=1000, final_iterations=7, device="cpu", log=lambda _t: None,
                              progress=lambda i, n: progress.append((i, n)), refit=False)
    order = rep["order"]
    assert rep["method"] == "independent" and order[0] == "c00" and sorted(order) == sorted(photos)
    # camera k of the visit order: seed + 97 k (one attempt each: nothing reframed)
    assert seeds == [1000 + 97 * k for k in range(1, len(order))]
    views = tmp_path / "out" / "views"
    assert (views / "00_c00_clean.png").exists()
    for k, v in enumerate(order[1:], start=1):
        for kind in ("input", "qwen", "clean"):
            assert (views / f"{k:02d}_{v}_{kind}.png").exists(), (k, v, kind)
    assert calls["support"]["restore_outside"] is True and calls["support"]["seed"] == 1000
    opt = calls["optimise"]
    assert opt["chain"] == order and opt["init_ply"] == filled and opt["iterations"] == 7
    assert opt["freeze_geometry"] and not opt["train_kept"] and set(opt["edits"]) == set(order)
    assert all(opt["edits"][u].shape == (256, 256, 3) for u in order)     # at the cameras' own size
    assert not filled.exists()                                               # the intermediate is removed
    assert calls["finish"]["done"] == order and calls["finish"]["region_at"]("c00").shape == (128, 128)
    assert rep["ply"] == str(tmp_path / "out" / "independent_scene.ply")
    assert (tmp_path / "out" / "independent_report.json").exists()
    assert progress[-1][0] == progress[-1][1]


def test_independent_fill_refit_at_the_cleans_size(tmp_path, monkeypatch):
    """The default refit: the optimiser gets the cameras, photos, edits and
    masks at the cleans' size (here 512 against the cameras' 256), the kept
    Gaussians train, the discs' geometry at 0.3, and no finishing."""
    from pathlib import Path

    from plyfile import PlyData, PlyElement

    import spede_core.progressive_fill as pf

    cams = ring(n=8)
    obj = np.random.default_rng(0).uniform(-0.3, 0.3, (400, 3))
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    v = np.zeros(len(obj), dtype=[(n, "<f4") for n in names])
    for i, k in enumerate("xyz"):
        v[k] = obj[:, i]
    v["rot_0"] = 1.0
    source = tmp_path / "source.ply"
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(source)
    photos = {c.view_id: np.full((256, 256, 3), 120, np.uint8) for c in cams.views}
    calls = {}
    filled = tmp_path / "out" / "support_surface.ply"

    def support(**kw):
        filled.write_bytes(b"ply")
        return {"ply": filled, "surface": {"discs": 3, "disc_spacing": 0.01}, "disc_means": np.zeros((3, 3)),
                "disc_ramp": np.ones(3), "protect": np.zeros((0, 3)), "top": None, "top_tiles": [], "Kh": None}

    def optimise(**kw):
        calls["optimise"] = kw
        Path(kw["out_ply"]).write_bytes(b"ply")
        return {"after": {}, "constraint_stats": {"births": {"drift": {"p50": 0.02}}}}

    monkeypatch.setattr(pf, "_support_surface", support)
    monkeypatch.setattr(pf, "multiview_optimize", optimise)
    monkeypatch.setattr(pf, "_finish_surface", lambda final, **kw: calls.setdefault("finish", kw))
    rep = pf.independent_fill(source_ply=source, start_ply=tmp_path / "depth_patch_scene.ply", cameras=cams,
                              hero="c00", hero_clean=np.full((64, 64, 3), 100, np.uint8), photos=photos,
                              selections={}, object_ids=np.arange(len(obj)), centre=np.zeros(3),
                              clean=lambda *_a: None, clean_single=lambda image, seed: image, out_dir=tmp_path / "out",
                              size=512, seed=1000, final_iterations=7, device="cpu", log=lambda _t: None)
    opt = calls["optimise"]
    order = rep["order"]
    assert "finish" not in calls
    assert opt["train_kept"] and not opt["freeze_geometry"] and opt["protect_kept"]
    assert (opt["geo_lr_births"], opt["geo_lr_kept"], opt["births_anchor"]) == (0.3, 0.0, 1e-2)
    assert all(c.width == 512 and c.height == 512 for c in opt["cameras"].views)
    assert all(opt["edits"][u].shape == (512, 512, 3) for u in order)
    assert all(opt["photos"][u].shape == (512, 512, 3) for u in photos)
    for u in order:
        assert opt["masks"][u].shape == (512, 512) and opt["masks"][u].sum() > 0
    assert rep["refit"]["target_size"] == 512 and rep["refit"]["mask_grow_px"] == 12
    assert rep["constraint_stats"]["births"]["drift_spacings"]["p50"] == 2.0


def test_row_scaled_step_defaults_and_frozen_rows():
    """Scale 1 everywhere is the plain Adam step; scale 0 leaves those rows
    unchanged; between, the step is that fraction (Adam's moments do not
    depend on the learning rate)."""
    import inspect

    import torch

    from spede_core.multiview_fill import _row_scaled_step, multiview_optimize

    sig = inspect.signature(multiview_optimize).parameters
    assert (sig["geo_lr_births"].default, sig["geo_lr_kept"].default, sig["births_anchor"].default,
            sig["protect_kept"].default) == (1.0, 1.0, 0.0, False)

    def run(scale, steps=3):
        torch.manual_seed(0)
        P = {"means": torch.nn.Parameter(torch.randn(6, 3)), "dc": torch.nn.Parameter(torch.randn(6, 3))}
        target = torch.randn(6, 3)
        opt = torch.optim.Adam([{"params": [P["means"]], "lr": 0.1}, {"params": [P["dc"]], "lr": 0.05}])
        start = {k: v.detach().clone() for k, v in P.items()}
        for _ in range(steps):
            loss = ((P["means"] - target) ** 2).sum() + ((P["dc"] + target) ** 2).sum()
            opt.zero_grad()
            loss.backward()
            if scale is None:
                opt.step()
            else:
                _row_scaled_step(opt, P, {"means": scale})
        return start, {k: v.detach() for k, v in P.items()}

    start, plain = run(None)
    _s, ones = run(torch.ones(6, 1))
    assert torch.allclose(plain["means"], ones["means"], atol=1e-6) and torch.equal(plain["dc"], ones["dc"])
    rows = torch.tensor([0.0, 1, 0, 1, 0.3, 0])[:, None]
    _s, part = run(rows)
    frozen = rows[:, 0] == 0
    assert torch.equal(part["means"][frozen], start["means"][frozen])
    assert torch.allclose(part["means"][~frozen & (rows[:, 0] == 1)], plain["means"][~frozen & (rows[:, 0] == 1)],
                          atol=1e-6)
    _s, plain1 = run(None, steps=1)
    _s, part1 = run(rows, steps=1)
    assert torch.allclose(part1["means"][4] - start["means"][4], 0.3 * (plain1["means"][4] - start["means"][4]),
                          atol=1e-6)
    assert torch.equal(part["dc"], plain["dc"])                       # unscaled parameters untouched


def test_cameras_at_scales_projection_exactly():
    """512 -> 1024: a point projects to exactly twice its 512 px coordinates
    (principal point included), at the same depth."""
    from spede_core.geometry_utils import project_to_pixels
    from spede_core.progressive_fill import cameras_at

    K512 = np.array([[796.49, 0, 256.25], [0, 798.5, 256.0], [0, 0, 1]], np.float32)
    cams = CameraSet([CameraView(view_id="a", viewmat=look_at((3, 1, 1.5)), intrinsic=K512, width=512, height=512)])
    big = cameras_at(cams, 1024)
    cam = big.by_id["a"]
    assert (cam.width, cam.height) == (1024, 1024) and cams.by_id["a"].width == 512
    pts = np.random.default_rng(1).uniform(-0.5, 0.5, (20, 3))
    u0, z0 = project_to_pixels(pts, cams.by_id["a"].viewmat, cams.by_id["a"].intrinsic)
    u1, z1 = project_to_pixels(pts, cam.viewmat, cam.intrinsic)
    assert np.allclose(u1, 2 * u0, atol=1e-4) and np.allclose(z1, z0)
