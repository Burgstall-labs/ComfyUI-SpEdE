# Third-party components (SpEdE Local node pack)

The AGPL-3.0 license of this pack covers only the parts written by Burgstall
Labs (see [NOTICE](NOTICE)). Everything below keeps its own license. This is an
engineering summary checked on 2026-10-08 and updated on 2026-10-28, not legal
advice: read each license before use, and recheck versions before a release.

## Code dependencies (installed by the user, not vendored)

| Component | Used for | License |
|---|---|---|
| ComfyUI | host application, native SAM3 / Qwen-Image / TextGenerate / MoGe nodes | GPL-3.0 |
| gsplat (Nerfstudio) | Gaussian rasterizer, contributor maps | Apache-2.0 |
| PyTorch | runtime | BSD-style |
| NumPy, SciPy | arrays, geometry | BSD-3-Clause |
| plyfile | PLY I/O | BSD-style |
| Pillow | images | HPND (MIT-CMU) |
| FastAPI | SpEdE UI server | MIT |
| uvicorn | SpEdE UI server | BSD-3-Clause |
| pydantic | SpEdE UI request models | MIT |
| httpx | tests of the UI server | BSD-3-Clause |
| comfy-sdk, comfy-api-proxy (optional) | Comfy SDK v2 runner | see each package |
| pycolmap (optional) | `geometry_cli.py import-colmap` | see the package |

## Downloaded at run time by the SpEdE UI (not vendored)

| Component | Used for | License |
|---|---|---|
| three.js 0.170.0 | 3D views; fetched from jsDelivr into `<runs>/viewer` on first start | MIT |
| GaussianSplats3D 0.4.7 (`@mkkellogg/gaussian-splats-3d`) | splat viewer; fetched the same way | MIT |
| Google Fonts: Jost, IBM Plex Mono | UI typography; the HTML pages load them from fonts.googleapis.com | SIL Open Font License 1.1 |

## Models and weights (downloaded by the user, never distributed here)

| Model | Used for | License | Notes |
|---|---|---|---|
| SAM 3.1 (Meta) | multiview segmentation; every workflow that selects | SAM License (Meta) | Commercial use allowed with use restrictions (no military, ITAR, nuclear or espionage use). |
| Qwen-Image 2.1 + Qwen3-VL 8B text encoder + VAE | Qwen3-VL names the selection in the hero check; Qwen-Image 2.1 repairs the view in the depth fill and the planar patch, cleans the hero in all-around stage 1, and edits in Replace staging | Qwen-Image 2.1 Research License ("Qwen RESEARCH LICENSE AGREEMENT") | Noncommercial research/evaluation only without a separate grant. Downloading the weights yourself does not give commercial rights. |
| Qwen-Image 2.1 turbo | all-around fill (independent and progressive); Replace draws the new object | same Qwen-Image 2.1 Research License | as above |
| MoGe-3 ViT-L (Microsoft), via Comfy-Org/MoGe | depth of the repaired view (depth fill, all-around) | MIT (model card) | MoGe code (microsoft/MoGe) is MIT; its DINOv2 backbone is Apache-2.0. |
| TripoSplat (VAST), via VAST-AI/TripoSplat (https://huggingface.co/VAST-AI/TripoSplat; code https://github.com/VAST-AI-Research/TripoSplat) | Replace: the new object as Gaussians (ComfyUI's native TripoSplat nodes) | MIT | The official README: "TripoSplat code and weight models are released under the MIT License." |
| DINOv3 ViT-H+ (Meta), `dino_v3_vit_h.safetensors` from the same repository | Replace: TripoSplat's image encoder | DINOv3 License (Meta), https://ai.meta.com/resources/models-and-libraries/dinov3-license/ | Commercial use allowed with use restrictions (no ITAR, military, nuclear or espionage use); redistribution must carry the licence. |
| FLUX.2 VAE (Black Forest Labs), `flux2-vae.safetensors` from the same repository | Replace: TripoSplat's latent space | Apache-2.0 | Only the autoencoder; the FLUX.2 [dev] transformer is under the FLUX Non-Commercial License and is not used. |
| MiniMax H3 ref2va, video/audio VAEs, Qwen3-VL 32B encoder, via Comfy-Org/MiniMax-H3 | experimental orbit only (all-around stage 2) | minimax-h3-community-license-agreement (as named on the model card) | Not needed for the product path. |
| MiniMax H3 DMD ref2va 8-step turbo LoRA (pruned), via drbaph/MiniMax-H3-Turbo-Lora-ComfyUI | experimental orbit only | card tag Apache-2.0; derivative of MiniMax H3 | The card asks users to also follow the licences of the original projects. |
| CrossView-Warp LoRA v1_3500 (Cseti), via Cseti/MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1 | experimental orbit only | MiniMax H3 Community License Agreement (per the card) | |

The Extract workflow needs only SAM 3.1. The SDK runner refuses to submit any
graph that loads a Qwen model (hero check, depth fill, planar patch,
all-around, Replace, Replace staging) without an explicit
`--accept-qwen-research-license`; the UI requires
`accept_qwen_research_license: true` in its config.

## Data

No scenes are bundled. `examples/make_demo_fixture.py` generates a procedural,
self-owned synthetic scene. If you use public research scenes (for example
Mip-NeRF 360 or pretrained Inria 3DGS models), their own terms apply; the Inria
Gaussian-Splatting license is noncommercial research and evaluation only.

Scenes used in development and shown in the demo video, **not distributed
here**:

| Data | Shown as | Credit | Notes |
|---|---|---|---|
| Mip-NeRF 360 (*bonsai*, *kitchen*) | development scenes, example outputs, demo video | Barron, Mildenhall, Verbin, Srinivasan, Hedman: "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields", CVPR 2022, https://jonbarron.info/mipnerf360/ | Research data; check the dataset terms before any commercial use. |
| Tanks and Temples (*Truck*) | demo video (Extract) | Knapitsch, Park, Zhou, Koltun: "Tanks and Temples: Benchmarking Large-Scale Scene Reconstruction", ACM Transactions on Graphics 36(4), SIGGRAPH 2017, https://www.tanksandtemples.org/ | Research data; check the dataset terms before any commercial use. |
| Pretrained 3DGS models of those scenes (Inria, https://github.com/graphdeco-inria/gaussian-splatting) | source splats | Kerbl, Kopanas, Leimkühler, Drettakis: "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023 | Gaussian-Splatting License (Inria / MPII): noncommercial research and evaluation only. |

The images in `examples/outputs/` are renders and model outputs of edited
Mip-NeRF 360 scenes (pretrained Inria models, Qwen-Image 2.1, MoGe-3). They
are not covered by the AGPL license of this pack; they keep the terms of those
scenes and models and are included for illustration only.

## Outputs

An edited scene carries the restrictions of every model and input that produced
it. The AGPL license of this code does not clear them.
