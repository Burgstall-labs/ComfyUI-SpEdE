# Depth-aware Fill v2

> Status (23 Oct 2026): GPU-verified on bonsai and kitchen 17–18 Oct; the UI's default Remove + Patch (`patch_mode: depthfill`).

This is a local Comfy workflow; the verified October 9 planar
Remove + Patch remains unchanged as the fallback. GPU-validated on bonsai and kitchen on
17 October 2026 after the residue carving below.

## 17 October 2026: residue carving (GPU-verified)

First GPU runs of the 16 October version both failed closed: bonsai at the
alignment gate (50.1 % inliers), kitchen at the patch (8 birth pixels).
Two findings drove the changes:

- After removal the hole is not empty. Only 73 (bonsai) and 84 (kitchen)
  selected pixels were uncovered; the rest is residue: the support surface
  under the object that no camera saw, stacked in several layers, plus a
  lower layer that renders dark from the hero direction. v2 kept all of it
  as "known background", so it could neither fill nor fix the dark spot.
- MoGe 3 compresses the jump between depth layers (bonsai: tabletop in front
  of a wall), and the support plane fitted to Gaussian centres lies ~4 %
  behind the rendered surface (kitchen), which displaced the planar patch.

Changes (`spede_core/depth_fill.py`, node `SpEdE_LocalPatchFromDepth`):

1. `align_depth_to_scene(depth_model="auto")`: a monotone (binned medians +
   PAV) map is used when the affine one fails or explains clearly less of the
   ring; the plane prior is scaled onto the rendered ring surface first.
2. `max_observed_weight` + `carve_residue_ids`: carve the Gaussians in the
   hole that no capture camera of the original scene saw (any depth), and
   the ones in front of the clean surface inside the object's bounding box.
   Replacing the whole visible layer is an option (`visible_layer`), off:
   it removed correctly seen surface and degraded the side views.
3. `create_depth_patch`: births also where the remaining render disagrees
   with the cleaned image (colour, 5 px median) and is not in front of the
   clean surface, or lies behind it where carving exposed something; radii
   capped at 2.5 stride footprints; no births within a stride of depth edges
   of the clean surface (not at the selection border).

| Scene | Depth fit | Carved | Births | G3 MAE | H1 | Planar 9 Oct G3 |
|---|---|---|---|---|---|---|
| kitchen DSCF0669 | affine 84 %, plane corrected -3.8 % | 126 331 | 2 317 | 0.052 | 4/4 | 0.06 (displaced) |
| bonsai DSCF5565 | monotone 67 % | 21 321 | 456 | 0.038 | 4/4 | 0.048 (dark spot) |

Same hero masks, labels and seed as the 9 October UI runs; ComfyUI jobs
56 s and 29 s on the RTX 5090. The UI uses this workflow with
`"patch_mode": "depthfill"` in `competition/studio/config.json`.

## 18 October 2026: stride 1, shadow halo, footprint discs (GPU-verified)

- Births every pixel (`pixel_stride` 1 in the workflow; node default 2,
  minimum 1; cap 200 000 births). Stride 4 → 2 → 1 on kitchen: G3 0.052 →
  0.045 → 0.041 (harness).
- `shadow_halo`: SAM selects the object, not its shadow; Qwen removes both,
  and the shadow stayed in the splat as a dark seam 4-24 px outside the
  selection. Pixels within `shadow_halo_px` (24) of the selection that the
  cleaned image brightens, in a region touching the selection, lose their
  Gaussians (centre there, not behind the rendered surface) and get births
  on the aligned depth, so the shadow also goes from the side cameras.
- Birth discs follow the pixel footprint on the surface (tangent and
  in-plane across axes). Round discs of the larger extent smeared the kitchen
  mat's ribs into a flat, pale film on the grazing tabletop. Depth outside
  the valid region is filled from the nearest valid pixel (a constant fill
  made every border birth's frame jump), and each axis is capped at 1.5x
  the median of the births around it.

UI runs (studio → ComfyUI, same projects as above; ComfyUI jobs 61 s and
41 s; results in `competition/examples/outputs/`):

| Scene | Births | Halo px | G3 MAE | H1 |
|---|---|---|---|---|
| kitchen | 43 880 | 6 888 | 0.035 | pass |
| bonsai | 9 350 | 2 105 | 0.033 | pass |

Still visible: the halo band is softer than its surroundings (about half the
local texture), and the grey/orange mat boundary from Qwen does not agree
with what the side cameras saw.

The UI's compare view now names the splat format explicitly: the
cache-busting `?g=` query hid the `.splat` extension and the patched pane
never loaded (it spun forever).

## Implemented graph

1. Original camera images → native SAM 3.1 multi-view selection → remove source Gaussians.
2. Qwen-Image 2.1 edits the hero original image, with the removed-scene render as reference.
3. The edited RGB goes to native LoadMoGeModel + MoGeInference (MoGe 3 ViT-L).
4. The SpEdE camera-FOV node gives MoGe the real calibrated field of view.
5. SpEdE reads raw MOGE_GEOMETRY depth/validity (never the normalized colorful MoGeRender preview).
6. SpEdE robustly aligns positive camera-Z to known source depth OUTSIDE the selected mask
   using an annular ring (at least 80 samples, 52% inliers, 5.5% max median error).
7. When a trustworthy support plane exists, its geometry clamps monocular bulges.
8. Only missing pixels inside the selection become new 3D Gaussians. Already known
   opaque background and foreground remain unchanged.
9. Original-camera Gaussian re-render goes through G3; independent capture
   cameras go through H1 outside-mask locality QA.

New workflow:
competition/workflows/spede_local_sam31_qwen21_moge3_depthfill_api.json

Original planar fallback (unchanged):
competition/workflows/spede_local_sam31_qwen21_patch_api.json

## User-provided weights

ComfyUI/models/geometry_estimation/moge_3_vitl_fp16.safetensors

Download from official Comfy-Org checkpoints after reviewing model terms.
An up-to-date ComfyUI build with native LoadMoGeModel and MoGeInference nodes
is required; the earlier verified ComfyUI 0.37.0 may need upgrading.
The model weights are NOT distributed inside the SpEdE repo.

Native source:
https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_moge.py
https://github.com/Comfy-Org/docs/blob/main/tutorials/utility/moge.mdx

## Local Comfy SDK v2

Set COMFY_BASE_URL=http://127.0.0.1:8189 with comfy-api-proxy running.
Examples (run from repository root with Comfy's Python environment):

~~~bash
python -m competition.local_sdk depthfill \
 --source-ply /path/source.ply --camera-targets /path/cameras.npz \
 --hero-view camera_0 --object-text "chair" --job-name depth1 \
 --comfy-root /path/ComfyUI --prompt-file /tmp/depth1.json

python -m competition.local_sdk depthfill \
 --source-ply /path/source.ply --camera-targets /path/cameras.npz \
 --hero-view camera_0 --object-text "chair" --job-name depth1 \
 --accept-qwen-research-license --submit
~~~

The alignment node saves both a numeric aligned_depth.npz and a
visualization, plus alignment_report.json. Do not use the preview as numeric
depth. The result is still a limited image-conditioned 3D patch, NOT proven
general 6DOF scene reconstruction.

## Test gates

- Run real MoGe ViT-L and Qwen inference on RTX 5090.
- Validate anchor registration and reported G2 depth alignment.
- Verify no Gaussian is birthed where the actual scene already provides background.
- Inspect hero G3 and 2+ independent neighboring cameras in 3D.
- Compare old planar vs depth-aware result on the bonsai tabletop and kitchen.
- Do not infer truth of previously occluded surfaces from H1 outside-mask QA alone.
- Local Qwen 2.1 weights remain research/noncommercial without a separate license.
- The public competition subset must not include AccGS or proprietary legacy code.

## Replace staging added

An optional two-Qwen scene-aware replacement-preparation workflow is now
available, alongside CPU-tested object/mask/shadow extraction and scene bbox
placement prior code. It does NOT automatically run TripoSplat or prove G4
registration. See [Replace staging](REPLACE_V1.md) for the exact boundary.

## Follow-up Replace stage

After depthfill passes: Qwen scene-aware insertion → SAM masks new object →
RGBA crop + separate shadow/contact layer → official TripoSplat Comfy model →
register world-space position and scale from old Gaussian bbox or support-plane
ray intersection → multi-view G4 proof. This stage is not part of Fill v2.
Official TripoSplat: https://github.com/VAST-AI-Research/TripoSplat
