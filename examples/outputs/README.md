# Example outputs

Renders of our edited scenes; the captures and pretrained models are not redistributed.

The scenes are *bonsai* and *kitchen* from the Mip-NeRF 360 dataset
(Barron et al., "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance
Fields", CVPR 2022, https://jonbarron.info/mipnerf360/). The 3DGS models
are the pretrained Inria models (https://github.com/graphdeco-inria/gaussian-splatting).
Both keep their own terms.

Each folder holds one depth-fill run from the SpEdE UI (18 Oct 2026, RTX 5090,
ComfyUI 0.41.0):

- `qwen_edit.png`: Qwen-Image 2.1's repair of the hero view.
- `patched_render.png`: the edited scene rendered in the hero camera.
- `depth_preview.png`: the MoGe-3 depth after alignment to the scene.
- `gates.json`: the run's gates (G1, G2, G3, H1), copied as written.
- `timings.json`: the ComfyUI job time, births, G3 and H1 (see `docs/DEPTH_FILL_V2.md`).

Each folder also holds one all-around (progressive) run from the SpEdE UI
(23 Oct 2026, RTX 5090, ComfyUI 0.41.0), downscaled losslessly to stay under 1 MB:

- `progressive_sheet.png`: the progressive fill in other cameras.
- `hero_clean.png`: Qwen's 1024 px hero clean, downscaled to 640 px.
- `timings.json`, block `allround`: time, cameras cleaned, births, fit MAE (see `docs/ALLROUND_FILL.md`).
