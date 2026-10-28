# All-around fill

> Status (23 Oct 2026): the independent fill is the product path (UI 'Make it hold from every direction', `allround_mode: independent`; workflow `spede_local_allround_progressive_api.json`, `method` independent, after the lean stage 1 `spede_local_allround_hero_api.json`). The progressive fill stays selectable (`method` progressive, `allround_mode: progressive`). The H3 orbit path (stage 1 + stage 2) is experimental and not in the UI.

The depth fill (DEPTH_FILL_V2.md) repairs the hole an object leaves in one
camera, the hero. It looks right there, but on kitchen its error was 0.10 in
other cameras, three times the hero's 0.034. The all-around fill makes the
same hole hold from every direction. It builds on a finished depth-fill run.

## Process

Stage 1 of the orbit path (`workflows/spede_local_allround_stage1_api.json`;
the independent and progressive fills run only its first two steps,
`spede_local_allround_hero_api.json`, which saves `hero_clean.png` with the
plan):

1. **Plan** (`SpEdE · AllRoundStart`). Every real camera is scored for the
   surface the hero mask selects (`view_rank`, as the UI ranks cameras): how
   much of it is unoccluded, in frame and large. Scoring the whole removed
   object instead (its underside and footprint too) chose far views whose
   depth and carving misled the fill: on kitchen 12 000 splats drifted
   against 3 400, and held-out views got worse. From the scores it picks:
   - the **chain**: the hero, then the best-seeing cameras at least 30°
     (then 20°, then 12°) from every camera already chosen, 10 in all;
   - the **orbit**: real cameras about 36° of azimuth apart around the
     object, from the hero back to the hero. The 260 frames are posed
     between them on an arc about the object, so every frame is near a
     real camera.

   It also renders the hero photo and the removed scene at 1024 px (the
   full-resolution capture photo's centre square, checked against the
   camera pack).
2. **Hero at 1024**: Qwen-Image 2.1 with the depth fill's prompt.
3. **Qwen chain** (`AllRoundQwenChain`). Each chain camera gets its own
   photo and the nearest camera already cleaned as its only reference.
   Several references made Qwen copy a reference's composition. When more
   than 5 % of the picture outside the selection changes (reframing), it
   tries the next seed, up to 3 attempts.
4. **Init** (`AllRoundInit`). Every chain camera's clean image gets its own
   MoGe-3 depth, aligned to the current scene. Its residue and shadow are
   carved, and births are added where the current render disagrees, but
   only where no earlier camera's births draw.
5. **Orbit warp** (`AllRoundWarp`). The hero's 1024 clean image is
   reprojected with its aligned depth into every orbit camera. Pixels the
   hero never saw are magenta, which is how the CrossView-Warp LoRA was
   trained.

Stage 1's Qwen chain, Init births and warp frames feed the orbit path; the
independent and progressive fills use only stage 1's planning
(`state.json`: run folder, source PLY, camera pack, segment report, hero,
label, centre, photo folder) and the 1024 hero clean. `AllRoundLoad` takes
either stage 1 for them (`require: hero_clean`); stage 2 still needs the
warp.

Stage 2 (`spede_local_allround_stage2_api.json`) runs as its own prompt.
In one prompt the Qwen and H3 models (about 56 GB) would sit in RAM
together.

6. **MiniMax H3** ref2va: 8-step turbo LoRA, SigmaShift 12/3, CrossView-Warp
   LoRA 0.8, res_multistep, 8 steps. It films the orbit from the warp,
   guided from frame 0, with the hero as reference image and still
   reference video.
7. **Optimise** (`AllRoundOptimize`).
   - Frames that slipped off their path are dropped (warp error ≥ 0.08 or
     a shift > 2 px at 512). The job stops if fewer than half stay.
   - Trainable: the births, plus the kept Gaussians inside the removed
     object's box (grown 15 %) that project into a chain camera's grown
     selection.
   - Inside the hole the targets are the chain images (hero ×2) and the
     video frames; outside, every real photo.
   - Training: 10 000 iterations of L1 0.8 + SSIM 0.2, with a slight push
     towards fully opaque or fully transparent splats.
   - Splats are anchored inside the box and near their start (penalty
     1e-2·((Δ/0.02·size)²)); those that still drift more than 8 % of the
     object size are pruned.
   - SH rest terms are frozen: web viewers draw only the base colour, so
     trained view-dependent terms showed there as haze.

Two cameras of the chain (never the hero) are held out and only evaluated.

## UI

Step 3 (Result) has **Make it hold from every direction**. It needs a
depth-fill result. By default (`allround_mode: independent`) it runs the
lean stage 1 and then the independent fill (about 10 min expected from the
kitchen experiment; not yet timed through the UI). `allround_mode:
progressive` runs the progressive fill instead: 16–24 min on an RTX 5090
(bonsai, kitchen, 23 Oct 2026), about 26 GB RAM (peak of the 21 Oct 2026
run; the 23 Oct runs ran under a 30 GB reservation). With `allround_mode:
orbit` it runs stages 1 and 2 (measured 19 Oct on kitchen: 3 min + 21 min).
Each mode runs in one GPU turn (gpu-turns kind `allround`, or
`allround_orbit` for the orbit). Afterwards the 3D compare shows the
all-around scene; the hero-only patch is kept as `hero_only.splat`.

Configure `photos` per scene (full-resolution capture photos, `<view>.JPG`)
for a sharp 1024 hero. Without it, the pack image is upsampled.

## Independent fill (product path, 23 October)

`AllRoundProgressive` with `method` independent (`independent_fill` in
`spede_core/progressive_fill.py`):

1. Every camera that sees the hole, except the hero, is cleaned on its own
   photo (the full-resolution photo's 1024 centre square, as for the hero)
   by Qwen-Image 2.1 turbo with the chain prompt minus its reference
   sentences, with no reference image. As in the progressive fill, a result
   that changed more than 5 % of the picture outside the hole is redone with
   the next seed (3 attempts); the edit is tone-matched to the photo along
   the seam and blended into it with the feathered hole region. The hero
   keeps its 1024 clean.
2. The support surface as in the progressive fill, with
   `support_surface_fill(restore_outside=True)` (below), coloured from the
   hero, then painted from above (`texture_from_top`, ring 40, the hero
   clean as reference) and sharpened in tiles, with the progressive node's
   `prompt`.
3. One optimisation (6000 iterations, geometry frozen, the feather band
   protected) against all the cleaned cameras.
4. The progressive fill's finishing: best-view texturing with the top view
   first (bonus 1000) and the tiles again, feather cross-fade, feather ramp
   reset, no harmonisation, `correct_disc_tone(structure=True)`.

No births and no per-camera optimisations: the result is
`independent_scene.ply`, moved to the node's `output_path`; per camera
`views/NN_<view>_{input,qwen,clean}.png`.

Why (kitchen, scratch experiment of 23 October, same depth-fill run as the
UI verification below):

- **Time**: about 6 min (support surface 26 s, top view and 16 tiles
  2.7 min, optimisation 85 s, finishing 71 s) plus 73 s for the 23
  independent Qwen cleans, against about 21 min for the progressive fill.
- **Ghost contours**: in side-by-side renders from the real cameras and
  orbit frames, the progressive result showed ghost contours on the orange
  mat; the independent result did not.
- **The dark red stain** on the orange mat beside the fill was not an edit
  or a floater. The support surface's carve reaches past the discs (the
  whole object's footprint, grown) and took the mat's light upper layer
  of Gaussians there (0.006–0.036 above the surface), exposing the darker,
  more opaque layer under it. `restore_outside` keeps the carved rows that
  are farther than 1.5 surface spacings from every disc (in the surface's
  plane) and less than 8.5 spacings above the disc surface. In spacings
  the thresholds follow the scene's scale: kitchen (spacing 0.0053) keeps
  7 509 rows, 98 % the same rows as the top-view rule used in the
  experiment (outside the top view's disc area + 3 px, below 0.045);
  bonsai (spacing 0.0066) keeps 767, coloured like the cloth around them
  (median RGB 0.38/0.32/0.43 against 0.36/0.29/0.40). Off by default in
  `support_surface_fill`; the progressive fill does not use it.

The progressive fill (below) remains selectable and unchanged.

## Verified through the UI (23 Oct 2026)

Both scenes ran end to end through the SpEdE UI: hero check, Remove (depth
fill), then **Make it hold from every direction** (progressive). ComfyUI
0.41.0, RTX 5090 (32 GB), pack exported at commit dc7766a. The UI reserved
30 GB RAM for the all-around step; the earlier 21 Oct progressive run peaked
at about 26 GB.

| | Kitchen (DSCF0669, "lego bulldozer") | Bonsai (DSCF5565, "bonsai tree and stand") |
|---|---|---|
| Hero check | one click, IoU 0.988 | six clicks, `click_mask_too_big` (IoU 0.57, the clicks also took the bicycle); the named mask was used |
| Remove (depth fill) | 70 s; 43 880 births, 127 514 carved | 41 s (`jobs.json`); 9 347 births, 21 409 carved (`comfy_patch.json`) |
| Gates | G1, G2 (affine, 84.3 % inliers, 1.7 % RMS error), G3 (coverage 1.0, RGB MAE 0.0347), H1 (4 views): all pass | G1, G2 (monotone, 66.8 % inliers, 4.5 % RMS error), G3 (coverage 1.0, RGB MAE 0.0329), H1 (4 views): all pass |
| All-around, wall-clock | 1460 s (stage 1 181 s, progressive about 21 min) | 950 s |
| Cameras cleaned | 24 (hero included) | 24 |
| Support discs | 510 148 | 165 856 |
| Births / trainable / pruned | 4 803 / 387 408 / 7 057 | 35 248 / 153 591 / 3 280 |
| Gaussians after | 1 882 230 | 1 250 655 |
| Fit MAE vs cleaned targets, hero / mean | 0.059 / 0.050 | 0.090 / 0.033 |
| ComfyUI output | 2.1 GB | 1.4 GB |

Sheets and hero cleans: `competition/examples/outputs/<scene>/`.

## What was tried and dropped (kitchen, scratchpad experiments, 18–19 Oct 2026)

| Variant | Result |
|---|---|
| Qwen chain only (no video) | right in the training cameras; held-out cameras limited by Qwen's inconsistency between views |
| 124-frame 512 / 1024 px H3 video, three cameras | follows the warp (121–124/124 usable), covers one side |
| **260-frame 1024 px orbit + 10 Qwen cameras** | straight mat edges from all eight held-out cameras that see the hole (kept) |
| 3DGS densification | no gain; needles from far angles |
| per-frame / smoothed pose refinement of the video frames | blurrier |
| free (unanchored) splats | white fuzz floating above the fill from low side views |

## Verified end to end in ComfyUI (19 Oct 2026, kitchen bulldozer)

Both workflows ran through the SpEdE nodes on the depth-fill run
`kitchen_kitchen_bulldozer_1009_131527_patch_132132`. Stage 1 took 3 min
(Qwen chain at about 2 s per camera with QwenImage21Cache) and stage 2 took
21 min (H3 about 19 min, the optimisation 80 s).

- 243 of 260 orbit frames were usable.
- 3 613 drifted splats were pruned.
- Held-out chain cameras: error 0.103 / 0.103 (the scratch run: 0.102 / 0.101).

Renders from all eight held-out capture cameras match the scratch run
accepted in the viewer. Run outside ComfyUI on the scratch run's inputs,
the optimiser reproduces that run (3 365 drifted, identical errors).

Two bugs found on the way:
- ComfyUI runs nodes under `inference_mode`: the optimiser re-enables
  autograd itself.
- The warp's 2x2 splat wrote the neighbour's colour last, which shifted the
  warp by a pixel.

## Progressive fill (bonsai, 19–20 October)

The orbit-video fill failed on bonsai, where the removed tree stands in the middle of the room on a cloth-covered stool:

- Hero-only warp: from behind, the frames were almost entirely magenta, and H3 invented another room. Only 12 of 260 frames passed the check.
- Warp rendered from the whole Init scene: H3 copied the warp pixel for pixel, including Init's mess.
- Real room outside the hole, magenta inside: H3 filled the tree's silhouette with a new object, and the frame check, which compares only known pixels, passed it.

Inpainting models compared on four bonsai cameras with the whole hole masked (`ab_test.py` style):

| Model | Result |
|---|---|
| LaMa, MAT | clean cloth; background structure (bike wheel) smeared |
| Qwen-Image 2512 + InstantX inpaint ControlNet | fills the silhouette with a new object |
| FLUX.2 Klein 9B | good in three views; invented a stool in the hero |
| MOSS-Image inpaint | invents objects (stools, blankets, a window) |
| **Qwen-Image 2.1 turbo edit (magenta)** | clean in all four views (used) |

The 2D fill was not the bottleneck. `AllRoundProgressive` (`spede_core/progressive_fill.py`) works one camera at a time:

1. The **support surface** (`surface_fill.py`) is built first. A quadratic height field is fitted to the real surface around the object's base. Residue above it is removed, together with large Gaussians below it whose extent reaches it. A grid of flat, overlapping discs then covers the object's whole footprint, limited to where the real surface is missing or thin and to the surface's convex extent. The discs take the surrounding surface's SH rest terms and their base colour from the hero clean image, with the SH terms subtracted.
2. Cameras are visited in angular order from the hero. At each camera:
   - the current scene is rendered;
   - Qwen 2.1 turbo cleans the hole region (uncovered pixels magenta, the nearest finished camera as reference);
   - the result's tone is matched to the real scene along the seam with a smooth, local, per-channel gain field;
   - births are added only where the render was empty, and nothing is carved;
   - a short optimisation runs against all finished cameras.
3. Only new Gaussians train, and only colour and opacity. Geometry is frozen.

Each fix came from measurement (`diag_fill.py`, `bright.py`; renders at orbit frames between the real cameras):

| Problem | Cause | Fix |
|---|---|---|
| fuzz | free optimisation turned the fill into a cloud (discs 58° off their local surface; hero births alone 0.3°) | geometry frozen |
| dark pits | footprint taken from the stand's legs only (hole under the pot); large dark Gaussians inside the stool | whole footprint; carve what reaches the surface |
| fill darker than the cloth | web viewers draw base colour only, and the discs had no SH terms; Qwen darkened its fill by 10–20 % per step | SH rest from the cloth; seam tone matching |
| speckled cloth around the fill | the real Gaussians' colours were trained against the cleaned images | births only |

Bonsai takes about 10 minutes for 24 cameras (Qwen turbo 3–6 s per camera).

Last of all, `harmonise_disc_colours` gives the discs the base-colour field of the real surface around them. That field is interpolated over the disc area as a membrane from the seam, and the discs' texture amplitude is scaled to match the surrounding cloth (bonsai: × 0.2). Before this, web viewers showed the fill evenly bright, about as bright as the cloth on the lit front, with crisp grey speckles: from behind it stood out as a pale patch.

The disc layer also extends a **feather band** past the fill's edge, over the real surface (about 5 % of the footprint's size). Its opacity ramps from full to zero across the band, and the real surface under it is kept. Because the optimisation trains opacity, `set_disc_feather` sets the ramp again at the end. On kitchen the uncovered fraction at orbit frames fell from 1.5 % to 1.35 %, close to the hero-only fill's 1.3 %, and the hard outline at oblique views largely went away.

Optimised against some twenty cleaned images that each draw their own detail, the discs learned the average, which was a blur. After the final optimisation, `texture_discs_from_views` therefore colours each disc from the clean images that see it best:

- The score is facing × pixels per unit, and the hero counts 1.5×.
- Discs are depth-tested against the rendered scene and must fall inside that view's cleaned region.
- Weights are `(score / best)^8`, so neighbouring discs mostly agree on one view and switch smoothly.

Harmonisation and the feather follow.

After the feather, `correct_disc_tone` matches the fill to its surroundings in the renders. In each view, `fill_tone_target` builds a target from the surroundings' local mean colour, interpolated over the fill, and adds the fill's texture scaled to the surroundings' texture strength. Each disc's base colour then moves by the facing-weighted mean difference at its pixel, in two passes. On bonsai, best-view texturing had left a lighter, high-contrast knit patch, and this toned it down.

Committing the texture after every step (`commit_texture`, now off by default) was tried on 21 October and was worse. The editor keeps what it is shown, so each pass's sampling blur carried into the next view and compounded: bonsai's knit became a flat dark blob, and kitchen's uncovered fraction rose from 1.34 % to 1.81 %.

**Top-view texture (21 October, current default, `top_texture=True`).** The per-view edits each drew the kitchen mats' edges and the wood gap between them in a different place, and every blend of them smeared. The fix decides the support surface's layout once, in one image:

1. `top_camera` places a virtual camera straight above the discs, at the real cameras' median distance.
2. `texture_from_top` renders the scene from it, marks the disc area (minus the feather band) magenta, and Qwen paints it with the surroundings as context and the hero as reference. Qwen continued the mats' edges and the gap cleanly.
3. The discs take their colour from that image, both at the start, so the progressive views render the settled layout, and at the end with a 1000× bonus. The other views colour only what the top view does not see.

With the top view in use, `harmonise_disc_colours` is skipped. It, like the default `correct_disc_tone`, replaces the fill's low frequencies with the surroundings interpolated over it, which turned the mats' edges and gap into a glassy smear.

`correct_disc_tone(structure=True)` instead scales each view's render by a smooth gain field measured along the seam (`match_seam_tone_field`), which keeps the structure.

Results:
- Kitchen (prog6q): the gap and both mats' edges are continuous from every orbit view.
- Bonsai (prog14q): the top edit was darker than the cloth seen from the side, and the seam gain brought it to the surroundings' brightness, so the outline is faint in most views.

**Seam: wandering band, colour cross-fade (21 October).** The disc layer's feather band, where it fades out over the real surface, still showed as the seam. Two changes:

- **Width varies along the outline** (`support_surface_fill(feather_jitter=0.6)`). The band's width follows smooth noise between 0.4× and 1.6× `feather` (features about twice its width), so the seam is not an even offset of the object's outline, which the eye picks out. The distance and width are interpolated at each disc. Before, they were per 2 cm raster cell, which drew the edge as a staircase. Discs in fill cells stay opaque, since the real surface under them is carved.
- **Colour fades too** (`crossfade_feather_to_surface`, `progressive_fill(feather_crossfade=True)`). After the final texturing, the scene is rendered from the top camera with the discs hidden. Each band disc's base colour becomes `ramp × own + (1 − ramp) × that render`. With the opacity ramp alone, the band showed the fill's weave half-transparent over a real weave that did not line up: a ghosted double pattern ringing the fill.

The first run with both (kitchen prog10, 21 October) barely changed the seam, for two reasons:

- **The optimiser pruned the band.** Of 118 000 band discs, 41 000 survived. Splats under 5 % opacity are pruned as weak, and the band is faint by design, so the fill kept a hard edge. `multiview_optimize(protect=...)` now never prunes the band's discs.
- **A dark stain lies on the real mat around the fill.** It is either a real stain or the removed object's shadow. The band faded into it, so it drew the seam from every side. `texture_from_top(ring=40)` now repaints the whole disc area, feather band included, plus a 40 px (at 1024) ring of real surface past it. The weight is 1 over the discs and falls to 0 at the ring's end. `recolour_surface_under` fades the real Gaussians seen there to the painted colours with the same weight; only Gaussians at the surface's depth are touched, not those above or below it.
