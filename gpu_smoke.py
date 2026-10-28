#!/usr/bin/env python3
"""SpEdE LOCAL GPU smoke: actual gsplat render, contributors, and 3D extraction.

Uses spede_core exclusively (no AccGS). Run inside Comfy's CUDA venv.
This is NOT run by CPU CI. Optional mask-report causes a full 3D selection
and source-splat object PLY extraction, but does not call Qwen or SAM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from plyfile import PlyData

from spede_core.ply import GaussianPly
from spede_core.renderer import CameraSet, GsplatRenderer


def smoke(source_ply:Path, target_npz:Path, *, view_id:str="",
          report_path:Path|None=None, hero_mask:Path|None=None,
          output_dir:Path=Path("output/spede-local-gpu"))->dict:
    import torch
    import gsplat
    from gsplat import rasterize_to_indices_in_range
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA GPU is required for contributor smoke")
    source=GaussianPly.load(source_ply)
    cameras=CameraSet.from_camera_pack(target_npz)
    view_id=view_id or cameras.views[0].view_id
    renderer=GsplatRenderer(source_ply=source.path,cameras=cameras,device="cuda")
    rgb,alpha,depth=renderer.render_rgbd(view_id,edited=False)
    weights=renderer.contribution_map(view_id,edited=False,blend=True)
    assert rgb.shape[:2]==alpha.shape==depth.shape
    assert weights.gaussian_ids.shape[:2]==alpha.shape
    if not np.any(alpha>0.01):
        raise RuntimeError("No visible source Gaussians")
    indices=weights.gaussian_ids
    if not np.any(indices>=0):
        raise RuntimeError("Contributor map contains no real Gaussian IDs")
    if int(indices.max())>=len(source.vertex) or int(indices.min())<-1:
        raise RuntimeError("Contributors outside source Gaussian rows")
    output_dir=Path(output_dir).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    summary={
        "cuda_gpu":torch.cuda.get_device_name(0),
        "torch_version":torch.__version__,
        "gsplat_version":getattr(gsplat,"__version__","unknown"),
        "contributor_api":getattr(rasterize_to_indices_in_range,"__name__",str(rasterize_to_indices_in_range)),
        "source_gaussians":len(source.vertex),
        "camera_count":len(cameras.views),
        "render_shape":list(rgb.shape),
        "nonempty_alpha_pixels":int(np.count_nonzero(alpha>0.01)),
        "nonempty_contributors":int(np.count_nonzero(indices>=0)),
        "object_extracted":False,
        "source_api_parity":"not tested against legacy runtime",
    }
    if report_path is not None:
        if hero_mask is None:
            raise ValueError("--hero-mask required when using --segment-report")
        from spede_core.selection import run_selection
        from spede_core.object_extract import extract_object
        from types import SimpleNamespace
        outcome=run_selection(
            source.path,Path(target_npz),Path(report_path),Path(hero_mask),
            output_dir/"selection",device="cuda",footprint_cleanup=False,
        )
        facade=SimpleNamespace(
            manifest=SimpleNamespace(source_ply=str(source.path)),
            cameras=cameras, renderer=renderer,
        )
        result=extract_object(
            facade,object_ids_path=Path(outcome["object_ids"]),
            segment_report=Path(report_path),hero_mask_path=Path(hero_mask),
            out_dir=output_dir/"extract",
            surface_rescue=False,depth_clean=False,
        )
        subset=PlyData.read(str(result["object_ply"]))["vertex"].data
        with np.load(outcome["object_ids"],allow_pickle=False) as stored:
            ids=stored["object_ids"]
        if not np.array_equal(subset,source.vertex[ids]):
            raise AssertionError("Source Gaussian property/order mismatch")
        summary.update(
            object_extracted=True,selected_gaussians=int(len(ids)),
            extraction_qa=result.get("k1"),
            selection_g1=outcome.get("gates",{}).get("removal_residual"),
            object_ply=result["object_ply"],
        )
    (output_dir/"gpu_smoke_report.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-ply",required=True,type=Path)
    parser.add_argument("--camera-targets",required=True,type=Path)
    parser.add_argument("--hero-view",default="")
    parser.add_argument("--segment-report",type=Path)
    parser.add_argument("--hero-mask",type=Path)
    parser.add_argument("--out",type=Path,default=Path("output/spede-local-gpu"))
    args=parser.parse_args()
    print(json.dumps(smoke(
        args.source_ply,args.camera_targets,view_id=args.hero_view,
        report_path=args.segment_report,hero_mask=args.hero_mask,
        output_dir=args.out
    ),indent=2,default=str))


if __name__=="__main__":
    main()
