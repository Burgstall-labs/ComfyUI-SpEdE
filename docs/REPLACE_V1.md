# Replace — a new object where the selected one stands

> Status: in the UI as "Replace with…" since 27 Oct 2026 (the beta label
> was dropped the same day; the limits stay on the card); v2
> ("inverse extract", size modes) the same day. Workflow
> competition/workflows/spede_local_replace_api.json, builder
> local_sdk.build_replace_graph, nodes SpEdE_LocalReplaceBase,
> SpEdE_LocalObjectCutout and SpEdE_LocalPlaceGenerated, geometry in
> spede_core/replace_place.py.

Replace is an **inverse extract**: the new object goes into the source scene
without exactly the extracted object, the set SpEdE_LocalExtract keeps
(selection without the removal footprint, surface rescue, depth clean and the
support trim at the contact line). The mat round the old object stays the
source scene. No Remove is needed: the card is on the selection check next to
Remove and Extract, and on the Result page. One ComfyUI job of about a
minute (a few more the first time, see 0):

0. Without a current extract (none, or older than the selection's
   inputs/hero_mask.png) build_replace_graph puts the Extract nodes in front
   (SAM 3.1 in every camera by the old object's name, the selection, Extract;
   node ids x3…x50, sharing the scene, camera and SAM loaders). Its results
   become the project's Extract (extract/, extract_session/), so the next
   Replace (and the Extract card) reuse it.
1. SpEdE_LocalReplaceBase: the base (source without the object rows, the old
   contact footprint filled, see "The fill") and the picture Qwen draws into:
   a Remove's cleaned hero (comfy/qwen_edit.png) when the project has one,
   else the base rendered in the hero camera.
2. Qwen-Image 2.1 turbo (8 Euler steps on the turbo sigmas, cfg 1,
   TextEncodeQwenImage21 at 1024) draws the new object: `<image1>` is that
   picture, `<image2>` the original hero photo "for position and size only",
   optional `<image3>` a reference picture the user added. In the Match and
   Larger modes the prompt also asks for the size ("about as big as the
   {old} was: it fills the same space…").
3. SAM 3.1 finds the object in Qwen's picture by the user's words (without a
   leading article); SpEdE_LocalObjectCutout fills its holes and keeps the
   largest region.
4. Native TripoSplat turns the cut-out into Gaussians (SplatToFile3D PLY,
   Graphdeco, SH degree 0).
5. SpEdE_LocalPlaceGenerated stands them in the scene:
   - the support plane of the old object (a robust plane through the
     unselected source Gaussians in a ring round its footprint, world up from
     the cameras);
   - yaw about the plane normal by silhouette IoU against Qwen's mask, front
     and back told apart by colour against Qwen's picture;
   - base on the plane, the horizontal position where Qwen drew it
     (silhouette centroid) but at most `max_shift` (default 0.25) of the old
     footprint's size from its centre;
   - the scale by the **size** mode (below);
   - the old object's leftovers go (see "The removal set"): Replace removes
     more than the extract, because a new object covers the spot;
   - the part of the old contact footprint the new object leaves bare is
     filled; source rows near the plane there go (the mat in the old object's
     shadow, which Extract leaves behind);
   - a procedural contact shadow: flat dark Gaussians from the blurred
     footprint of the object's lowest 12 % (at most 30 % opaque).

**On a Remove (28 Oct 2026).** When the project has a finished Remove, the
base is its result scene as it is: the all-around fill
(comfy_allround.json `allround_ply`), else the hero patch
(comfy_patch.json `patched_ply`). The object is gone and its spot filled
there, so Replace takes nothing out, fills nothing and drops no leftovers
(`remove_scene_ply` on both nodes); the source and the extract still give
the support plane and the old object's size. Placement and the contact
shadow are the same. The Remove fill laid flat into the footprint (the
earlier way) read as a lighter blotchy patch on kitchen, worse than the
Remove itself.

Every run starts from the Remove's result or, without one, the source scene
and the extract, so Replace again or "Try again (new seed)" swaps the new
object, never stacks a second one; the earlier result is set aside
(`_previous_replace_*`). Remove sets a Replace
aside. The all-around fill is refused after a Replace (it would show the
scene without the new object).

## Size

`size` on the card and the node:

- **Match original** (default): the new object's dominant extent becomes
  the removed object's. The dominant extent is the larger of the height
  (1–99 %) and the footprint's width, taken as sqrt(length × width) (2–98 %
  along the footprint's own axes: the side of a square of the same area).
  That is what reads as "as big as": a vase as tall as the old object was
  big, a bowl with the old footprint's area. Matching heights leaves a flat
  object tiny where a long one stood (beta: the bowl at 0.41× the
  bulldozer's height); matching the footprint's length makes a round bowl
  1.7× the long, narrow bulldozer's footprint (1.66 × 0.75 on kitchen);
  matching footprints makes a vase a tower. A footprint-dominant object (its
  footprint at least as wide as it is tall) is grown further to cover the old
  footprint's length and width, by at most 1.25×; a tall one is not, the
  fill covers the rest. The new object is scaled about the base position the
  natural fit found (where Qwen drew it, the shift cap applied): refitting the
  centroid at a bigger size would pull it towards the camera.
- **Natural**: the beta behaviour, Qwen's size (the silhouette's area).
- **Larger**: 1.3 × Match.

The natural fit is computed and reported in every mode
(`height_natural_vs_removed`), so one run shows both Qwen's size and the
chosen one. When the 3D object is scaled away from Qwen's drawing, the hero
render differs from Qwen's picture (bigger object, its shadow and the fill
round it SpEdE's, not Qwen's); the 3D scene is the result.

## The fill

replace_place.replace_fill, on a grid of 1/100 of the old footprint's size
on the support plane:

- the region: the old object's lower 35 % rasterised, closed, holes filled,
  grown by 3 % of the footprint size; minus where the new object touches the
  plane (its own lowest 10 %, closed, one cell in, so the fill reaches under
  its rim and belly: under a vase's belly the ground shows from a low
  camera);
- with a Remove on record (its all-around scene, else its depth fill): the
  Remove's fill Gaussians in the region up to 15 % of the old height above
  the plane, laid flat on it (0.2 % above), with their own colours (Qwen's
  repaint of the hero);
- every cell they leave bare (or all of them without a Remove): one flat
  disc, coloured from the surface in a ring 6–30 % of the footprint's size
  out (right next to it the mat lies in the old object's shadow), only
  Gaussians on the surface (within 1 % of the old height), each cell taking
  the nearest ring cell's colour, blurred by 1.5 cells. Plain, deterministic,
  untextured.

## The removal set

The extract is the object Extract keeps, not everything the old object left
in the scene. Kitchen (27 Oct 2026, the vase from DSCF0813): dark and yellow
streaks in front of the vase were opaque, elongated bulldozer splats inside
its footprint at 24–42 % of its height that the selection never took (22
opaque rows above the band, 210 rows in all); the dark splats at the vase's
base were the tracks' contact rows on the plane (the support trim's hard cut
and unselected dark rows, 1 798 opaque dark rows in the footprint), kept
under the vase's belly because the fill stopped at the new object's lower
35 %. Replace drops, on top of the extract, in the old object's **volume**
(its whole footprint at every height, closed, grown by 3 % of the footprint
size, from 10 % of its height below the plane to 105 % above):

- everything above the band (5 % of the old height), whatever its colour:
  missed parts and floaters inside the old object;
- in the band and below the plane: everything inside the filled region; in
  the rest of the volume (under an overhang, under the new object's foot)
  what is not the support's colour, as support_trim decides it (nearer the
  old object's colours round it than the ring's, or further from the ring's
  than 4 × the ring's own scatter; the ring 6–30 % of the footprint size out
  of the volume).

Nothing beyond the volume changes: on kitchen no row 6–30 % out of the old
footprint is dropped, and every dropped row is within 3 % of its size. The
report gives `volume_rows_dropped`, `leftover_contact_rows_dropped` and
`hole_rows_dropped`.

## Limits

- One hero camera decides the object's shape; its back is TripoSplat's
  guess. The yaw is reliable for objects with a visible front; a symmetric
  object can face either way.
- The old object's cast shadow beyond its footprint stays in the source
  scene (baked into the capture); only the contact under it is replaced.
- Without a Remove the fill is a flat colour: it reads as a smooth patch on
  a textured surface where the new object does not cover the old footprint.
- Qwen-Image 2.1 Research License applies to local inference. TripoSplat
  code and weights are MIT per its official README.

The earlier staging workflow spede_local_replace_prepare_api.json
(SpEdE_LocalPrepareReplace, SpEdE_LocalRegisterSplat: a bbox-prior merge
without yaw, grounding or shadow) is kept for reference.
