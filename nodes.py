"""SpEdE competition nodes: local-first, AccGS-free, Comfy-compatible.

The node pack has no mandatory cloud model API. In the primary workflow
SAM3.1 and Qwen Image 2.1 are native Comfy model nodes; SpEdE owns only
multiview Gaussian selection, geometry, and source/heldout QA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import logging
import shutil
import numpy as np

from spede_core.ply import GaussianPly, remove_rows
from spede_assets import (
    allowed_local_path, asset_fingerprint, mask_directory_fingerprint,
    masks_from_comfy_batch, instances_from_comfy_batch, masks_from_directory,
    read_selected_ids, write_segment_report,
)


SCENE="SPEDE_SCENE"
CAMERAS="SPEDE_CAMERAS"
SELECTION="SPEDE_SELECTION"
REPORT="SPEDE_REPORT"
LOG=logging.getLogger("SpEdE")


@dataclass(frozen=True)
class LocalScene:
    ply: Path
    count: int
    sh_degree: int


@dataclass(frozen=True)
class LocalCameras:
    targets_path: Path
    cameras: object


@dataclass(frozen=True)
class LocalSelection:
    scene: LocalScene
    cameras: LocalCameras
    object_ids_path: Path
    report_path: Path
    hero_mask_path: Path
    hull_report_path: Path
    device: str


def _input_path(value):
    path=allowed_local_path(Path(value).expanduser().resolve())
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _output_dir(value, default):
    raw=str(value or "").strip()
    try:
        import folder_paths
        root=Path(folder_paths.get_output_directory()).resolve()
    except ImportError:
        root=(Path.cwd()/"output").resolve()
    out=Path(raw) if raw else Path("spede")/default
    out=allowed_local_path(out if out.is_absolute() else root/out,output=True)
    out.mkdir(parents=True,exist_ok=True)
    return out


def _output_ply(path, default):
    value=str(path or "").strip()
    desired=Path(value) if value else Path("spede")/default
    if desired.suffix.lower()!=".ply":
        desired=desired.with_suffix(".ply")
    return _output_dir(str(desired.parent),"scenes")/desired.name


def _mask_path(selection:LocalSelection, view_id:str) -> Path:
    data=json.loads(selection.report_path.read_text())
    view_id=view_id.strip() or str(data["hero_view_id"])
    if view_id==data["hero_view_id"]:
        return selection.hero_mask_path
    row=data["views"].get(view_id)
    if not row:
        raise ValueError(f"No selection mask in camera {view_id!r}")
    return _input_path(row["mask_path"])


def _mask_array(selection:LocalSelection,view_id:str):
    from PIL import Image
    with Image.open(_mask_path(selection,view_id)) as im:
        mask=np.asarray(im.convert("L"))>127
    return mask


def _tensor_image(image):
    import torch
    arr=np.asarray(image,np.float32)
    if arr.dtype!=np.float32:
        arr=arr.astype(np.float32)
    if arr.max()>1:
        arr=arr/255.0
    return torch.from_numpy(np.ascontiguousarray(arr[None]))


def _tensor_mask(mask):
    import torch
    return torch.from_numpy(np.asarray(mask,np.float32)[None].copy())


def _renderer(scene: LocalScene, cameras: LocalCameras, device: str):
    from spede_core.renderer import GsplatRenderer
    return GsplatRenderer(source_ply=scene.ply,cameras=cameras.cameras,device=device)


class LocalLoadScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"source_ply":("STRING",{"default":""})}}
    @classmethod
    def IS_CHANGED(cls,source_ply):
        return asset_fingerprint(source_ply)
    RETURN_TYPES=(SCENE,"STRING")
    RETURN_NAMES=("scene","metadata_json")
    FUNCTION="load"
    CATEGORY="SpEdE Local/IO"

    def load(self,source_ply):
        ply=GaussianPly.load(_input_path(source_ply))
        scene=LocalScene(ply.path,len(ply.vertex),ply.max_sh_degree)
        return scene,json.dumps({
            "source_ply":str(ply.path),"gaussians":scene.count,
            "sh_degree":scene.sh_degree,"accgs":False
        })


class LocalLoadCameras:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"camera_targets_npz":("STRING",{"default":""})}}
    @classmethod
    def IS_CHANGED(cls,camera_targets_npz):
        return asset_fingerprint(camera_targets_npz)
    RETURN_TYPES=(CAMERAS,"STRING")
    RETURN_NAMES=("cameras","camera_ids_json")
    FUNCTION="load"
    CATEGORY="SpEdE Local/IO"

    def load(self,camera_targets_npz):
        from spede_core.renderer import CameraSet
        from spede_assets import inspect_camera_pack
        path=_input_path(camera_targets_npz)
        inspect_camera_pack(path)
        cameras=CameraSet.from_camera_pack(path) # Native camera NPZ reader
        return LocalCameras(path,cameras),json.dumps([v.view_id for v in cameras.views])


class LocalCameraImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"cameras":(CAMERAS,)}}
    RETURN_TYPES=("IMAGE","STRING")
    RETURN_NAMES=("images","view_ids_json")
    FUNCTION="images"
    CATEGORY="SpEdE Local/IO"

    def images(self,cameras):
        import torch
        from spede_core.imageio import rgb_u8
        with np.load(cameras.targets_path,allow_pickle=False) as f:
            targets=f["targets"]
        data=np.stack([rgb_u8(a) for a in targets]).astype(np.float32)/255
        return torch.from_numpy(data),json.dumps([v.view_id for v in cameras.cameras.views])


class LocalMasksToReport:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "cameras":(CAMERAS,),
            "masks":("MASK",),
            "view_ids_json":("STRING",{"default":"[]"}),
            "hero_view_id":("STRING",{"default":""}),
            "output_subdir":("STRING",{"default":"spede/local/masks"}),
        },"optional":{
            "hero_mask_path":("STRING",{"default":""}),
            # SAM3_Detect's boxes with individual_masks on: then ``masks`` is
            # every view's instances and the selection picks among them.
            "boxes":("BOUNDING_BOX",{"forceInput":True}),
            "object_text":("STRING",{"default":""}),
        }}
    RETURN_TYPES=("STRING","STRING","STRING")
    RETURN_NAMES=("segment_report_json","hero_mask_png","report_json")
    FUNCTION="prepare"
    CATEGORY="SpEdE Local/Select"

    def prepare(self,cameras,masks,view_ids_json,hero_view_id,output_subdir,hero_mask_path="",
                boxes=None,object_text=""):
        labels=json.loads(view_ids_json)
        if not isinstance(labels,list) or not all(isinstance(x,str) for x in labels):
            raise ValueError("view_ids_json must be a JSON array of labels")
        instances=None
        if boxes is not None:
            instances=instances_from_comfy_batch(masks,boxes,labels,cameras.cameras.by_id)
            data={v:(inst>0.5).any(axis=0).astype(np.float32) for v,inst in instances.items()}
            # Without the user's own hero mask the hero keeps SAM's best
            # instance alone, as before instances were asked for.
            hero_inst=instances.get(hero_view_id.strip())
            if hero_inst is not None and len(hero_inst):
                data[hero_view_id.strip()]=(hero_inst[0]>0.5).astype(np.float32)
        else:
            data=masks_from_comfy_batch(masks,labels,cameras.cameras.by_id)
        if hero_mask_path.strip():
            # A user's own hero selection (e.g. Studio's click-SAM mask) wins
            # over the text-prompted SAM mask in that one view.
            from PIL import Image
            camera=cameras.cameras.by_id[hero_view_id.strip()]
            with Image.open(_input_path(hero_mask_path.strip())) as im:
                hero=np.asarray(im.convert("L"))>127
            if hero.shape!=(int(camera.height),int(camera.width)):
                raise ValueError("hero_mask_path size differs from the hero camera")
            if not hero.any():
                raise ValueError("hero_mask_path is empty")
            data[hero_view_id.strip()]=hero.astype(np.float32)
        path,hero,report=write_segment_report(
            cameras.cameras.by_id,data,hero_view_id=hero_view_id,
            output_dir=_output_dir(output_subdir,"masks"),minimum_voting_views=3,
            instances=instances,object_text=object_text
        )
        return str(path),str(hero),json.dumps(report)


class LocalSelectMultiView:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "scene":(SCENE,),"cameras":(CAMERAS,),
            "segment_report_json":("STRING",{"default":""}),
            "hero_mask_png":("STRING",{"default":""}),
            "output_subdir":("STRING",{"default":"spede/local/selection"}),
            "include_footprint":("BOOLEAN",{"default":False}),
            "device":("STRING",{"default":"cuda"}),
        }}
    RETURN_TYPES=(SELECTION,REPORT,"STRING")
    RETURN_NAMES=("selection","report","object_ids_npz")
    FUNCTION="select"
    CATEGORY="SpEdE Local/Select"

    def select(self,scene,cameras,segment_report_json,hero_mask_png,
               output_subdir,include_footprint,device):
        from spede_core.selection import run_selection
        report=_input_path(segment_report_json)
        hero=_input_path(hero_mask_png)
        result=run_selection(
            scene.ply,cameras.targets_path,report,hero,
            _output_dir(output_subdir,"selection"),device=device,
            footprint_cleanup=bool(include_footprint)
        )
        # The selection's own report: each camera's mask cut to the
        # instances that hold the hero's object.
        sel=LocalSelection(
            scene,cameras,_input_path(result["object_ids"]),_input_path(result["segment_report"]),hero,
            _input_path(Path(result["object_ids"]).parent/"object_hull_report.json"),device
        )
        return sel,json.dumps(result),str(sel.object_ids_path)


class LocalExtract:
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "selection":(SELECTION,),
            "output_subdir":("STRING",{"default":"spede/local/extract"}),
            "surface_rescue":("BOOLEAN",{"default":True}),
            "depth_clean":("BOOLEAN",{"default":True}),
        },"optional":{
            # Drop the table-mat fringe the selection takes at the contact
            # line (removal wants it, an extracted object does not).
            "support_trim":("BOOLEAN",{"default":True}),
            # Drop small clusters clearly apart from the object (another
            # thing that projected inside the masks, e.g. a railing).
            "floater_cleanup":("BOOLEAN",{"default":True}),
        }}
    RETURN_TYPES=("STRING",REPORT)
    RETURN_NAMES=("object_ply","report_json")
    FUNCTION="extract"
    CATEGORY="SpEdE Local/Extract"

    def extract(self,selection,output_subdir,surface_rescue,depth_clean,support_trim=True,
                floater_cleanup=True):
        from spede_core.object_extract import extract_object
        from spede_core.renderer import GsplatRenderer
        with np.load(selection.object_ids_path,allow_pickle=False) as f:
            if "column_applied" in f.files and bool(f["column_applied"]):
                raise ValueError("Extraction requires selection without removal footprint cleanup")
        engine=SimpleNamespace(
            manifest=SimpleNamespace(source_ply=str(selection.scene.ply)),
            cameras=selection.cameras.cameras,
            renderer=_renderer(selection.scene,selection.cameras,selection.device),
        )
        result=extract_object(
            engine, object_ids_path=selection.object_ids_path,
            segment_report=selection.report_path,
            hero_mask_path=selection.hero_mask_path,
            out_dir=_output_dir(output_subdir,"extract"),
            surface_rescue=bool(surface_rescue),depth_clean=bool(depth_clean),
            support_trim=bool(support_trim),floater_cleanup=bool(floater_cleanup)
        )
        return str(result["object_ply"]),json.dumps(result)


class LocalRemove:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "selection":(SELECTION,),
            "output_path":("STRING",{"default":"spede/local/removed_scene.ply"}),
        }}
    RETURN_TYPES=(SCENE,REPORT)
    RETURN_NAMES=("removed_scene","report_json")
    FUNCTION="remove"
    CATEGORY="SpEdE Local/Remove"

    def remove(self,selection,output_path):
        ids=read_selected_ids(selection.object_ids_path,selection.scene.count)
        path=_output_ply(output_path,"removed_scene.ply")
        result=remove_rows(selection.scene.ply,ids,path)
        result["g1_selection"]=json.loads(selection.hull_report_path.read_text()).get("gates",{}).get("removal_residual",{})
        return LocalScene(path,int(result["remaining_gaussians"]),selection.scene.sh_degree),json.dumps(result)


class LocalMaskView:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"selection":(SELECTION,),"view_id":("STRING",{"default":""})}}
    RETURN_TYPES=("MASK","STRING")
    RETURN_NAMES=("mask","view_id")
    FUNCTION="view"
    CATEGORY="SpEdE Local/IO"

    def view(self,selection,view_id):
        data=json.loads(selection.report_path.read_text())
        view_id=view_id.strip() or data["hero_view_id"]
        return _tensor_mask(_mask_array(selection,view_id)),view_id


class LocalRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "scene":(SCENE,),"cameras":(CAMERAS,),
            "view_id":("STRING",{"default":""}),
            "device":("STRING",{"default":"cuda"}),
        }}
    RETURN_TYPES=("IMAGE","MASK","STRING")
    RETURN_NAMES=("render","alpha","view_id")
    FUNCTION="render"
    CATEGORY="SpEdE Local/Render"

    def render(self,scene,cameras,view_id,device):
        if view_id not in cameras.cameras.by_id:
            raise ValueError(f"Missing camera {view_id!r}")
        rgb,alpha,_depth=_renderer(scene,cameras,device).render_rgbd(view_id,edited=False)
        return _tensor_image(rgb),_tensor_mask(alpha),view_id


class LocalPatchPlane:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "removed_scene":(SCENE,),
            "selection":(SELECTION,),
            "edited_image":("IMAGE",),
            "view_id":("STRING",{"default":""}),
            "output_path":("STRING",{"default":"spede/local/patch_scene.ply"}),
            "pixel_stride":("INT",{"default":5,"min":2,"max":64}),
            "ring_px":("INT",{"default":24,"min":3,"max":128}),
        },"optional":{
            "clear_contact":("BOOLEAN",{"default":True}),
        }}
    RETURN_TYPES=(SCENE,REPORT)
    RETURN_NAMES=("patched_scene","patch_report_json")
    FUNCTION="patch"
    CATEGORY="SpEdE Local/Patch"

    def patch(self,removed_scene,selection,edited_image,view_id,output_path,
              pixel_stride,ring_px,clear_contact=True):
        from spede_core.patch import (
            fit_view_support_plane, patch_support_plane, unobserved_contact_ids,
        )
        image=np.asarray(edited_image.detach().cpu() if hasattr(edited_image,"detach")
                         else edited_image,np.float32)
        if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]<3:
            raise ValueError("Qwen edit must contain exactly one IMAGE output")
        data=json.loads(selection.report_path.read_text())
        vid=view_id.strip() or data["hero_view_id"]
        if vid not in selection.cameras.cameras.by_id:
            raise ValueError("Unknown selected view")
        camera=selection.cameras.cameras.by_id[vid]
        mask=_mask_array(selection,vid)
        output=_output_ply(output_path,"patched_scene.ply")
        removed=GaussianPly.load(removed_scene.ply)
        plane,tolerance=fit_view_support_plane(removed,camera,mask,ring_px=int(ring_px))
        base=removed_scene
        contact={"enabled":bool(clear_contact),"cleared_gaussians":0}
        if clear_contact:
            ids=read_selected_ids(selection.object_ids_path,selection.scene.count)
            object_means=GaussianPly.load(selection.scene.ply).means[ids]
            cleared=unobserved_contact_ids(removed,object_means,plane,camera,mask,tolerance)
            contact["cleared_gaussians"]=int(len(cleared))
            if len(cleared):
                path=output.with_name(output.stem+"_contact_cleared.ply")
                remove_rows(removed_scene.ply,cleared,path)
                base=LocalScene(path,removed_scene.count-len(cleared),removed_scene.sh_degree)
        rgb,alpha,depth=_renderer(
            base,selection.cameras,selection.device
        ).render_rgbd(vid,edited=False)
        result=patch_support_plane(
            base.ply,camera,mask,image[0,:,:,:3],output,
            remaining_alpha=alpha,remaining_depth=depth,
            ring_px=int(ring_px),pixel_stride=int(pixel_stride),plane=plane,
        )
        result["contact_layer"]=contact
        report_path=output.with_suffix(".patch_report.json")
        report_path.write_text(json.dumps(result,indent=2)+"\n")
        return LocalScene(output,int(result["total_gaussians"]),removed_scene.sh_degree),json.dumps(result)


def _write_report(scene:LocalScene,name:str,report:dict)->Path:
    """QA reports live next to the judged scene so SDK/Studio callers can
    read them without parsing node outputs."""
    path=Path(scene.ply).with_suffix(f".{name}.json")
    path.write_text(json.dumps(report,indent=2,default=str)+"\n")
    return path


class LocalPatchQA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "patched_scene":(SCENE,),"selection":(SELECTION,),
            "target_image":("IMAGE",),
            "view_id":("STRING",{"default":""}),
        }}
    RETURN_TYPES=("BOOLEAN",REPORT,"IMAGE")
    RETURN_NAMES=("g3_passed","g3_report_json","actual_render")
    FUNCTION="judge"
    CATEGORY="SpEdE Local/QA"
    OUTPUT_NODE=True

    def judge(self,patched_scene,selection,target_image,view_id):
        from spede_core.patch import patch_image_metrics
        data=json.loads(selection.report_path.read_text())
        vid=view_id.strip() or data["hero_view_id"]
        reference=np.asarray(target_image.detach().cpu() if hasattr(target_image,"detach")
                             else target_image,np.float32)
        if reference.ndim!=4 or reference.shape[0]!=1:
            raise ValueError("Target image must contain exactly one frame")
        actual,alpha,_depth=_renderer(
            patched_scene,selection.cameras,selection.device
        ).render_rgbd(vid,edited=False)
        metric=patch_image_metrics(reference[0,:,:,:3],actual,_mask_array(selection,vid),alpha)
        metric["report_path"]=str(_write_report(patched_scene,"g3_report",metric))
        return {"ui":{"spede_report":[json.dumps(metric)]},
                "result":(bool(metric["passed"]),json.dumps(metric),_tensor_image(actual))}


class LocalSaveScene:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "scene":(SCENE,),
            "output_path":("STRING",{"default":"spede/local/final_scene.ply"}),
        }}
    RETURN_TYPES=("STRING",)
    RETURN_NAMES=("scene_ply",)
    FUNCTION="save"
    CATEGORY="SpEdE Local/IO"
    OUTPUT_NODE=True

    def save(self,scene,output_path):
        destination=_output_ply(output_path,"final_scene.ply")
        if scene.ply.resolve()!=destination.resolve():
            shutil.copy2(scene.ply,destination)
        return (str(destination),)




class LocalHeldoutQA:
    """Compare actual source and patched GS renders in independent capture views.

    This checks outside-object scene preservation, not true GT of the
    hallucinated pixels behind the removed object.
    """
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "source_scene":(SCENE,),
            "patched_scene":(SCENE,),
            "selection":(SELECTION,),
            "maximum_mean_abs":("FLOAT",{"default":0.03,"min":0.0,"max":1.0}),
            "maximum_changed_fraction":("FLOAT",{"default":0.07,"min":0.0,"max":1.0}),
            "max_views":("INT",{"default":4,"min":1,"max":16}),
        }}

    RETURN_TYPES=("BOOLEAN",REPORT)
    RETURN_NAMES=("preserves_heldout_views","heldout_report_json")
    FUNCTION="judge"
    CATEGORY="SpEdE Local/QA"

    def judge(self,source_scene,patched_scene,selection,
              maximum_mean_abs,maximum_changed_fraction,max_views):
        from spede_core.quality import outside_mask_preservation
        record=json.loads(selection.report_path.read_text())
        others=list(record["voting_views"])[:int(max_views)]
        source=_renderer(source_scene,selection.cameras,selection.device)
        patched=_renderer(patched_scene,selection.cameras,selection.device)
        cases=[]
        for view_id in others:
            baseline=source.render(view_id,edited=False)
            edited=patched.render(view_id,edited=False)
            mask=_mask_array(selection,view_id)
            score=outside_mask_preservation(
                baseline,edited,mask,
                maximum_mean_abs=float(maximum_mean_abs),
                maximum_changed_fraction=float(maximum_changed_fraction)
            )
            score["view_id"]=view_id
            cases.append(score)
        passed=len(cases)>=2 and all(row["passed"] for row in cases)
        report={
            "name":"H1 independent capture-view locality",
            "passed":bool(passed),
            "judged_views":len(cases),
            "views":cases,
            "criterion":"RGB outside the SAM selected-object mask remains source-like",
            "not_a_ground_truth_patch_completeness_test":True,
        }
        report["report_path"]=str(_write_report(patched_scene,"heldout_report",report))
        return {"ui":{"spede_report":[json.dumps(report)]},
                "result":(bool(passed),json.dumps(report))}


def _array(value):
    """A Comfy IMAGE/MASK tensor (or array) as numpy."""
    return np.asarray(value.detach().cpu() if hasattr(value,"detach") else value,np.float32)


class LocalViewImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"cameras":(CAMERAS,),"view_id":("STRING",{"default":""})}}
    RETURN_TYPES=("IMAGE","STRING")
    RETURN_NAMES=("image","view_id")
    FUNCTION="image"
    CATEGORY="SpEdE Local/IO"

    def image(self,cameras,view_id):
        from spede_core.imageio import rgb_u8
        labels=[v.view_id for v in cameras.cameras.views]
        view_id=view_id.strip() or labels[0]
        if view_id not in labels:
            raise ValueError(f"Missing camera {view_id!r}")
        with np.load(cameras.targets_path,allow_pickle=False) as f:
            target=f["targets"][labels.index(view_id)]
        return _tensor_image(rgb_u8(target)),view_id


class LocalFocusCrop:
    """The region around a selection with everything else dimmed: what a
    vision-language model (native Generate Text + Qwen3-VL) names."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "image":("IMAGE",),"mask":("MASK",),
            "size":("INT",{"default":448,"min":64,"max":2048}),
            "dim":("FLOAT",{"default":0.35,"min":0.0,"max":1.0,"step":0.05}),
        }}
    RETURN_TYPES=("IMAGE",)
    RETURN_NAMES=("crop",)
    FUNCTION="crop"
    CATEGORY="SpEdE Local/Select"

    def crop(self,image,mask,size,dim):
        from spede_core.hero_check import focus_crop
        img,m=_array(image),_array(mask)
        region=m.reshape(-1,*m.shape[-2:]).max(axis=0)
        return (_tensor_image(focus_crop(img[0],region,size=int(size),dim=float(dim))),)


class LocalObjectName:
    """A model's answer as a short object name for SAM's text prompt."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"text":("STRING",{"forceInput":True})}}
    RETURN_TYPES=("STRING",)
    RETURN_NAMES=("name",)
    FUNCTION="name"
    CATEGORY="SpEdE Local/Select"

    def name(self,text):
        from spede_core.hero_check import clean_name
        name=clean_name(text)
        if not name:
            raise ValueError(f"No object name in {text!r}")
        return (name,)


class LocalHeroConsistency:
    """Checks the hero view before the other cameras are searched by name:
    the clicked mask against what the name finds in the same image. Writes
    the chosen hero mask (MasksToReport's hero_mask_path), both masks, a
    preview and hero_check.json."""
    OUTPUT_NODE=True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "image":("IMAGE",),
            "click_mask":("MASK",),
            "name_masks":("MASK",),
            "name":("STRING",{"default":""}),
            "positive_coords":("STRING",{"default":"[]"}),
            "negative_coords":("STRING",{"default":"[]"}),
            "min_iou":("FLOAT",{"default":0.6,"min":0.05,"max":1.0,"step":0.05}),
            "output_subdir":("STRING",{"default":"spede/local/hero"}),
        }}
    RETURN_TYPES=("MASK","STRING","STRING","BOOLEAN","IMAGE","STRING")
    RETURN_NAMES=("hero_mask","hero_mask_png","name","passed","preview","report_json")
    FUNCTION="check"
    CATEGORY="SpEdE Local/Select"

    def check(self,image,click_mask,name_masks,name,positive_coords,negative_coords,min_iou,output_subdir):
        from PIL import Image
        from spede_core.hero_check import compare_hero_masks, overlay
        img=_array(image)[0]
        click=_array(click_mask)
        click=click.reshape(-1,*click.shape[-2:]).max(axis=0)
        named=_array(name_masks)
        named=named.reshape(-1,*named.shape[-2:]) if named.size else np.zeros((0,*click.shape),np.float32)
        chosen,name_mask,report=compare_hero_masks(
            click,named,positive_coords,negative_coords,min_iou=float(min_iou))
        out=_output_dir(output_subdir,"hero")
        save=lambda m,n: Image.fromarray((np.asarray(m)>0.5).astype(np.uint8)*255).save(out/n)
        save(chosen,"hero_mask.png")
        save(click,"click_mask.png")
        if name_mask is not None:
            save(name_mask,"name_mask.png")
        else:
            (out/"name_mask.png").unlink(missing_ok=True)
        preview=overlay(img,click,chosen,name_mask)
        Image.fromarray(preview).save(out/"preview.png")
        report.update(name=str(name).strip(),hero_mask_png=str(out/"hero_mask.png"),
                      click_mask_png=str(out/"click_mask.png"),preview_png=str(out/"preview.png"),
                      name_mask_png=str(out/"name_mask.png") if name_mask is not None else None)
        (out/"hero_check.json").write_text(json.dumps(report,indent=2)+"\n")
        return {"ui":{"spede_report":[json.dumps(report)]},
                "result":(_tensor_mask(chosen),str(out/"hero_mask.png"),str(name).strip(),
                          bool(report["passed"]),_tensor_image(preview),json.dumps(report))}





@dataclass(frozen=True)
class LocalDepthProposal:
    aligned: object
    camera_id: str
    source_ply: Path


class LocalCameraFOV:
    """Horizontal FOV from actual capture intrinsics for native MoGe-3."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"cameras":(CAMERAS,),
                            "view_id":("STRING",{"default":""})}}
    RETURN_TYPES=("FLOAT",)
    RETURN_NAMES=("fov_x_degrees",)
    FUNCTION="fov"
    CATEGORY="SpEdE Local/Depth"

    def fov(self,cameras,view_id):
        import math
        if view_id not in cameras.cameras.by_id:
            raise ValueError(f"Unknown capture camera {view_id!r}")
        c=cameras.cameras.by_id[view_id]
        fx=float(c.intrinsic[0,0])
        if not math.isfinite(fx) or fx<=0:
            raise ValueError("Invalid camera focal length")
        return (math.degrees(2*math.atan(c.width/(2*fx))),)


class LocalAlignMoGeDepth:
    """Align native MOGE_GEOMETRY raw camera depth to the known scene."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "moge_geometry":("MOGE_GEOMETRY",),
            "removed_scene":(SCENE,),
            "selection":(SELECTION,),
            "view_id":("STRING",{"default":""}),
            "plane_prior":(["auto","require","none"],),
            "ring_px":("INT",{"default":32,"min":4,"max":256}),
            "output_subdir":("STRING",{"default":"spede/local/depth"}),
        },"optional":{
            # The cleaned image: refuse when it removed more than was
            # selected (spede_core.depth_fill.edit_spill).
            "edited_image":("IMAGE",),
            "max_spill_fraction":("FLOAT",{"default":0.10,"min":0.0,"max":10.0,"step":0.01}),
        }}
    RETURN_TYPES=("SPEDE_DEPTH","IMAGE",REPORT)
    RETURN_NAMES=("aligned_depth","depth_preview","alignment_report")
    FUNCTION="align"
    CATEGORY="SpEdE Local/Depth"

    def align(self,moge_geometry,removed_scene,selection,view_id,
              plane_prior,ring_px,output_subdir,edited_image=None,max_spill_fraction=0.10):
        from PIL import Image
        from spede_core.depth_fill import align_depth_to_scene, edit_spill
        from spede_core.geometry_utils import plane_depth_map
        from spede_core.patch import fit_view_support_plane

        if not isinstance(moge_geometry,dict) or "depth" not in moge_geometry:
            raise ValueError("MoGe must return raw MOGE_GEOMETRY with depth")
        info=json.loads(selection.report_path.read_text())
        vid=view_id.strip() or info["hero_view_id"]
        if vid not in selection.cameras.cameras.by_id:
            raise ValueError(f"Unknown camera {vid!r}")
        camera=selection.cameras.cameras.by_id[vid]
        prediction=_array(moge_geometry["depth"])
        if prediction.ndim!=3 or prediction.shape[0]!=1:
            raise ValueError("MoGe depth must contain one camera [1,H,W]")
        valid=None
        if moge_geometry.get("mask") is not None:
            valid=_array(moge_geometry["mask"])
            if valid.shape!=prediction.shape:
                raise ValueError("MoGe validity-mask dimensions differ")
            valid=valid[0]>0.5
        mask=_mask_array(selection,vid)
        rgb,alpha,scene_z=_renderer(
            removed_scene,selection.cameras,selection.device
        ).render_rgbd(vid,edited=False)
        out=_output_dir(output_subdir,"depth")
        spill_report=None
        if edited_image is not None:
            edited=_array(edited_image)[0,:,:,:3]
            spill=edit_spill(mask,rgb,edited)
            pixels=int(np.count_nonzero(spill))
            spill_report={"pixels":pixels,"fraction_of_selection":round(pixels/max(int(mask.sum()),1),4),
                          "max_fraction":float(max_spill_fraction)}
            if pixels:
                preview=np.clip(edited*255,0,255).astype(np.uint8).copy()
                preview[spill]=(0.4*preview[spill]+[153,0,153]).astype(np.uint8)
                Image.fromarray(preview).save(out/"edit_spill.png")
            if pixels>=max(1500,float(max_spill_fraction)*mask.sum()):
                raise RuntimeError(
                    f"Qwen-Image removed more than the selection: it changed {pixels} px outside "
                    f"the {int(mask.sum())}-px selection ({pixels/max(int(mask.sum()),1):.0%} of it; "
                    "magenta in edit_spill.png). "
                    "The 3D scene would keep what the picture lost. Select everything the name covers, "
                    "or name only the selected part.")
        prior=None
        plane_info={"mode":plane_prior,"fitted":False}
        if plane_prior!="none":
            try:
                plane,_tol=fit_view_support_plane(
                    GaussianPly.load(removed_scene.ply),camera,mask,
                    ring_px=int(ring_px))
                prior=plane_depth_map(camera.viewmat,camera.intrinsic,
                                      camera.width,camera.height,
                                      plane.normal,plane.offset)
                plane_info.update(fitted=True,ring=plane.ring)
            except (RuntimeError,ValueError) as e:
                plane_info["reason"]=str(e)
                if plane_prior=="require":
                    raise
        # The inputs of the fit, saved before it can fail closed: a refused
        # alignment is otherwise impossible to diagnose after the run.
        np.savez_compressed(out/"alignment_inputs.npz",
                            predicted_depth=prediction[0],
                            predicted_valid=(np.ones_like(mask) if valid is None else valid),
                            source_depth=scene_z,source_alpha=alpha,mask=mask,
                            plane_depth=(np.full(mask.shape,np.nan) if prior is None else prior))
        try:
            aligned=align_depth_to_scene(
                prediction[0],scene_z,alpha,mask,
                predicted_valid=valid,support_plane_depth=prior,
                ring_px=int(ring_px))
        except RuntimeError as e:
            if (prior is None or plane_prior=="require" or
                "disagrees with known planar" not in str(e)):
                raise
            plane_info["reason"]="Predicted depth disagreed with candidate plane; kept known ring anchors"
            aligned=align_depth_to_scene(
                prediction[0],scene_z,alpha,mask,
                predicted_valid=valid,ring_px=int(ring_px))
        report={**aligned.report,"view_id":vid,"plane_fit":plane_info,"edit_spill":spill_report,
                "source_ply":str(removed_scene.ply)}
        (out/"alignment_report.json").write_text(json.dumps(report,indent=2)+"\n")
        np.savez_compressed(out/"aligned_depth.npz",
                            camera_z=aligned.depth,valid=aligned.valid,
                            anchor_mask=aligned.anchor_mask)
        vals=aligned.depth[aligned.valid]
        lo,hi=np.percentile(vals,[2,98])
        preview=np.zeros_like(aligned.depth)
        preview[aligned.valid]=np.clip((hi-aligned.depth[aligned.valid])/max(hi-lo,1e-6),0,1)
        return (LocalDepthProposal(aligned,vid,removed_scene.ply),
                _tensor_image(np.repeat(preview[...,None],3,axis=-1)),
                json.dumps(report))


class LocalPatchFromDepth:
    """Create new 3D Gaussian geometry from aligned edited-image depth."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "removed_scene":(SCENE,),
            "selection":(SELECTION,),
            "edited_image":("IMAGE",),
            "aligned_depth":("SPEDE_DEPTH",),
            "view_id":("STRING",{"default":""}),
            "output_path":("STRING",{"default":"spede/local/depth_patch_scene.ply"}),
            "pixel_stride":("INT",{"default":2,"min":1,"max":64}),
            "clear_contact":("BOOLEAN",{"default":True}),
        },"optional":{
            # Replace the residue the removal leaves in the hole (see
            # spede_core.depth_fill.carve_residue_ids); off keeps v2's
            # fill-only-empty-pixels behaviour.
            "carve_residue":("BOOLEAN",{"default":True}),
            "visible_layer":("FLOAT",{"default":0.0,"min":0.0,"max":0.5,"step":0.01}),
            # The object's shadow just outside the selection, which Qwen
            # removed (spede_core.depth_fill.shadow_halo); 0 keeps it.
            "shadow_halo_px":("INT",{"default":24,"min":0,"max":128}),
            # Kept colours fade into the cleaned image over this band around
            # the births and selection, at 512 px (spede_core.depth_fill.
            # feather_seam_colours); 0 keeps the hard seam.
            "seam_feather_px":("INT",{"default":24,"min":0,"max":128}),
        }}
    RETURN_TYPES=(SCENE,REPORT)
    RETURN_NAMES=("patched_scene","patch_report_json")
    FUNCTION="patch"
    CATEGORY="SpEdE Local/Patch"

    def patch(self,removed_scene,selection,edited_image,aligned_depth,view_id,
              output_path,pixel_stride,clear_contact,carve_residue=True,visible_layer=0.0,
              shadow_halo_px=24,seam_feather_px=24):
        from spede_core.depth_fill import carve_residue_ids, carving_exposed, create_depth_patch, max_observed_weight, shadow_halo
        from spede_core.patch import fit_view_support_plane, unobserved_contact_ids
        if not isinstance(aligned_depth,LocalDepthProposal):
            raise TypeError("Expected registered SpEdE MoGe depth")
        info=json.loads(selection.report_path.read_text())
        vid=view_id.strip() or info["hero_view_id"]
        if vid!=aligned_depth.camera_id or removed_scene.ply!=aligned_depth.source_ply:
            raise ValueError("Depth and edited scene must come from the same camera and PLY")
        camera=selection.cameras.cameras.by_id[vid]
        mask=_mask_array(selection,vid)
        image=_array(edited_image)
        if image.ndim!=4 or image.shape[0]!=1 or image.shape[-1]<3:
            raise ValueError("Qwen output must be one edited RGB image")
        output=_output_ply(output_path,"depth_patch_scene.ply")
        base=removed_scene
        old_ids=read_selected_ids(selection.object_ids_path,selection.scene.count)
        # Source row of every row of the scene being edited (remove_rows keeps order).
        source_rows=np.setdiff1d(np.arange(selection.scene.count),old_ids)
        cleanup={"enabled":bool(clear_contact),"cleared_gaussians":0}
        if clear_contact:
            try:
                gs=GaussianPly.load(removed_scene.ply)
                plane,tol=fit_view_support_plane(gs,camera,mask)
                old_means=GaussianPly.load(selection.scene.ply).means[old_ids]
                to_clear=unobserved_contact_ids(
                    gs,old_means,plane,camera,mask,tol)
                if len(to_clear):
                    cleaned=output.with_name(output.stem+"_contact_cleared.ply")
                    remove_rows(removed_scene.ply,to_clear,cleaned)
                    base=LocalScene(cleaned,removed_scene.count-len(to_clear),
                                    removed_scene.sh_degree)
                    source_rows=np.delete(source_rows,to_clear)
                    cleanup["cleared_gaussians"]=int(len(to_clear))
            except (RuntimeError,ValueError) as e:
                cleanup["reason"]="No trusted contact cleanup: "+str(e)
        carve={"enabled":bool(carve_residue)}
        exposed=None
        halo=None
        if carve_residue:
            rgb_before,alpha_before,depth_before=_renderer(
                base,selection.cameras,selection.device).render_rgbd(vid,edited=False)
            if shadow_halo_px>0:
                halo=shadow_halo(mask,rgb_before,image[0,:,:,:3],halo_px=int(shadow_halo_px))
            views=[v.view_id for v in selection.cameras.cameras.views if not v.is_virtual]
            seen=max_observed_weight(
                _renderer(selection.scene,selection.cameras,selection.device),
                views,selection.scene.count)[source_rows]
            ids,carve=carve_residue_ids(
                GaussianPly.load(base.ply),
                GaussianPly.load(selection.scene.ply).means[old_ids],
                camera,mask,aligned_depth.aligned.depth,observed_weight=seen,
                visible_layer=float(visible_layer) if visible_layer>0 else None,
                halo=halo,halo_depth=depth_before)
            carve["enabled"]=True
            if len(ids):
                carved=output.with_name(output.stem+"_carved.ply")
                remove_rows(base.ply,ids,carved)
                base=LocalScene(carved,base.count-len(ids),removed_scene.sh_degree)
        remaining_rgb,alpha,source_z=_renderer(
            base,selection.cameras,selection.device
        ).render_rgbd(vid,edited=False)
        if carve_residue:
            exposed=carving_exposed(alpha_before,depth_before,alpha,source_z)
            carve["exposed_selected_pixels"]=int(np.count_nonzero(exposed&mask))
            carve["shadow_halo_px"]=int(shadow_halo_px)
            carve["shadow_halo_pixels"]=int(np.count_nonzero(halo)) if halo is not None else 0
        from spede_core.depth_fill import match_edit_tone
        # Which pixels are born is decided on Qwen's image as it came; the
        # births take its colours in the scene's exposure.
        toned,tone=match_edit_tone(image[0,:,:,:3],remaining_rgb,mask)
        feather=int(seam_feather_px)>0
        patched=output.with_name(output.stem+"_unfeathered.ply") if feather else output
        result=create_depth_patch(
            base.ply,camera,image[0,:,:,:3],aligned_depth.aligned,
            # Stride 1 at 512 px needs ~45k births on kitchen: the default
            # cap would thin whole rows out of the patch.
            source_z,alpha,patched,pixel_stride=int(pixel_stride),max_births=200000,exposed=exposed,
            remaining_rgb=remaining_rgb if carve_residue else None,
            halo=halo,halo_depth=depth_before if halo is not None else None,birth_rgb=toned)
        if feather:
            from spede_core.depth_fill import feather_seam_colours, write_dc
            from spede_core.visual_hull import scaled_dilation
            total=int(result["total_gaussians"])
            hero=_renderer(LocalScene(patched,total,removed_scene.sh_degree),
                           selection.cameras,selection.device)
            rendered,_a,_z=hero.render_rgbd(vid,edited=False)
            gs=GaussianPly.load(patched)
            dc,result["seam_feather"]=feather_seam_colours(
                gs.means,gs.dc[:,0],np.arange(total)>=int(result["source_gaussians"]),
                camera,mask,toned,rendered,max_observed_weight(hero,[vid],total),
                band_px=scaled_dilation(int(seam_feather_px),mask.shape),birth_px=int(pixel_stride))
            result["scene_ply"]=str(write_dc(gs,dc,output,"SpEdE seam feather: kept colours eased into the cleaned image"))
        else:
            result["seam_feather"]={"applied":False,"band_px":0}
        result["contact_layer"]=cleanup
        result["edit_tone"]=tone
        result["residue_carve"]=carve
        result["report_path"]=str(output.with_suffix(".patch_report.json"))
        Path(result["report_path"]).write_text(json.dumps(result,indent=2)+"\n")
        return (LocalScene(output,int(result["total_gaussians"]),
                           removed_scene.sh_degree),json.dumps(result))



class LocalPrepareReplace:
    """Scene-aware object SAM cutout, shadow candidate and scene-derived size."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "selection":(SELECTION,),
            "edited_image":("IMAGE",),
            "clean_background_image":("IMAGE",),
            "new_object_mask":("MASK",),
            "view_id":("STRING",{"default":""}),
            "output_subdir":("STRING",{"default":"spede/local/replacement"}),
            "scale_multiplier":("FLOAT",{"default":1.0,"min":0.25,"max":4.0}),
        }}
    RETURN_TYPES=("IMAGE","MASK",REPORT)
    RETURN_NAMES=("tripo_reference_rgb","object_alpha","placement_report_json")
    FUNCTION="prepare"
    CATEGORY="SpEdE Local/Replace"

    def prepare(self,selection,edited_image,clean_background_image,new_object_mask,
                view_id,output_subdir,scale_multiplier):
        from PIL import Image
        from spede_core.replace import image_object_assets,replacement_prior
        images=_array(edited_image)
        background=_array(clean_background_image)
        masks=_array(new_object_mask)
        if images.ndim!=4 or background.shape!=images.shape or len(images)!=1:
            raise ValueError("Replacement needs matching single-view edited and clean RGB")
        if masks.ndim!=3 or len(masks)!=1:
            raise ValueError("Replacement SAM mask must be one [1,H,W] mask")
        meta=json.loads(selection.report_path.read_text())
        vid=view_id.strip() or meta["hero_view_id"]
        if vid not in selection.cameras.cameras.by_id:
            raise ValueError("Unknown camera for replacement")
        cam=selection.cameras.cameras.by_id[vid]
        m=masks[0]>0.5
        out=_output_dir(output_subdir,"replacement")
        asset=image_object_assets(images[0,:,:,:3],background[0,:,:,:3],m,out)
        old_ids=read_selected_ids(selection.object_ids_path,selection.scene.count)
        placement=replacement_prior(selection.scene.ply,old_ids,cam,m,
                                    scale_multiplier=float(scale_multiplier))
        report={**asset,"placement":placement}
        path=out/"placement_prior.json"
        path.write_text(json.dumps(report,indent=2)+"\n")
        rgba=np.asarray(Image.open(asset["image_object_rgba"]).convert("RGBA"),
                        dtype=np.float32)/255
        reference=rgba[:,:,:3]*rgba[:,:,3,None]+(1-rgba[:,:,3,None])
        return (_tensor_image(reference),
                _tensor_mask(rgba[:,:,3]),json.dumps(report))


class LocalRegisterSplat:
    """Merge external Graphdeco-format generated object PLY at scene prior.

    Official TripoSplat inference/conversion to PLY occurs upstream and is
    NOT claimed to be included inside this node.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "background_scene":(SCENE,),
            "generated_object_ply":("STRING",{"default":""}),
            "placement_report_json":("STRING",{"default":""}),
            "output_path":("STRING",{"default":"spede/local/replaced_scene.ply"}),
        }}
    RETURN_TYPES=(SCENE,REPORT)
    RETURN_NAMES=("registered_scene","registration_report_json")
    FUNCTION="register"
    CATEGORY="SpEdE Local/Replace"

    def register(self,background_scene,generated_object_ply,
                 placement_report_json,output_path):
        from spede_core.replace import merge_tripo_ply
        path=_input_path(generated_object_ply)
        try:
            meta=json.loads(placement_report_json)
        except (ValueError,TypeError) as e:
            raise ValueError("placement_report_json must be valid placement JSON") from e
        prior=meta.get("placement",meta)
        output=_output_ply(output_path,"replaced_scene.ply")
        result=merge_tripo_ply(background_scene.ply,path,prior,output)
        result["output_type"]="registered_gaussian_splats_not_shadow_layer"
        return (LocalScene(output,int(result["result_gaussians"]),
                           background_scene.sh_degree),json.dumps(result))


class LocalObjectCutout:
    """SAM 3.1's mask of the inserted object, made whole: holes filled (SAM's
    "bowl" left the fruit out on kitchen) and only the largest region kept.
    Feeds TripoSplat (image + object mask) and SpEdE_LocalPlaceGenerated."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "image":("IMAGE",),"mask":("MASK",),
            "name":("STRING",{"default":"object"}),
        }}
    RETURN_TYPES=("IMAGE","MASK","IMAGE")
    RETURN_NAMES=("image","object_mask","cutout_on_white")
    FUNCTION="cut"
    CATEGORY="SpEdE Local/Replace"

    def cut(self,image,mask,name):
        from scipy import ndimage
        img,m=_array(image),_array(mask)
        if img.ndim!=4 or len(img)!=1:
            raise ValueError("Cutout needs exactly one image")
        m=m.reshape(-1,*m.shape[-2:]).max(axis=0)>0.5
        if m.shape!=img.shape[1:3]:
            raise ValueError(f"SAM mask {m.shape} does not match the image {img.shape[1:3]}")
        m=ndimage.binary_fill_holes(m)
        labels,count=ndimage.label(m)
        if count==0 or m.sum()<64:
            raise ValueError(f"SAM 3.1 found no {str(name).strip() or 'object'!r} in Qwen-Image's picture: "
                             "Qwen did not draw it, or under another name; try again with a new seed or other words")
        m=labels==1+int(np.argmax(ndimage.sum(m,labels,range(1,count+1))))
        rgb=img[0,:,:,:3]
        white=rgb*m[...,None]+(1-m[...,None])
        return _tensor_image(rgb),_tensor_mask(m.astype(np.float32)),_tensor_image(white)


def _replace_scene(scene,cameras,object_ply,remove_fill_ply):
    """Replace's inputs: the source scene, the extracted object's rows in it
    (Extract's object_only.ply, matched by position), the support plane they
    stood on and the Remove fill Gaussians if the project has a Remove."""
    from spede_core.geometry_utils import world_up_from_cameras
    from spede_core.replace_place import fill_rows,object_rows,ply_gaussians,support_plane
    source=GaussianPly.load(scene.ply)
    obj=GaussianPly.load(_input_path(object_ply))
    ids=object_rows(source.means,obj.means)
    if len(ids)<max(16,len(obj.vertex)//2):
        raise ValueError(f"{object_ply} is not an extract of {scene.ply}: only {len(ids)} of its "
                         f"{len(obj.vertex)} Gaussians are source rows")
    views=cameras.cameras
    up=world_up_from_cameras([np.linalg.inv(v.viewmat.astype(np.float64)) for v in views.views if not v.is_virtual])
    plane=support_plane(source.means,source.opacity_logits,ids,up)
    remove_fill=None
    if str(remove_fill_ply or "").strip():
        removed=GaussianPly.load(_input_path(remove_fill_ply))
        remove_fill=ply_gaussians(removed,np.flatnonzero(fill_rows(removed.means,source.means)))
    rgb=ply_gaussians(source).rgb
    return source,ids,plane,remove_fill,rgb


def _remove_scene(remove_scene_ply):
    """A Remove's result scene to build on (the all-around fill, else the
    hero patch), or None: the object is gone and its spot filled there, so
    Replace takes nothing out and fills nothing; the source and the extract
    only measure the old object (plane, footprint, size)."""
    if not str(remove_scene_ply or "").strip():
        return None
    return GaussianPly.load(_input_path(remove_scene_ply))


class LocalReplaceBase:
    """Replace's base before the new object: the source scene without
    exactly the extracted object, its contact footprint filled
    (spede_core.replace_place.replace_fill; from the Remove fill when there
    is one), or ``remove_scene_ply`` (a Remove's result) as it is, and the
    hero picture Qwen-Image draws into: ``hero_image_path`` (a Remove's
    cleaned hero) when given, else this base rendered."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "scene":(SCENE,),"cameras":(CAMERAS,),
            "view_id":("STRING",{"default":""}),
            "object_ply":("STRING",{"default":""}),
            "output_path":("STRING",{"default":"spede/local/replace_base.ply"}),
            "device":("STRING",{"default":"cuda"}),
        },"optional":{
            "remove_fill_ply":("STRING",{"default":""}),
            "hero_image_path":("STRING",{"default":""}),
            "remove_scene_ply":("STRING",{"default":""}),
        }}
    RETURN_TYPES=(SCENE,"IMAGE",REPORT)
    RETURN_NAMES=("base_scene","hero_image","report_json")
    FUNCTION="base"
    CATEGORY="SpEdE Local/Replace"

    def base(self,scene,cameras,view_id,object_ply,output_path,device,remove_fill_ply="",hero_image_path="",
             remove_scene_ply=""):
        from PIL import Image
        from spede_core.replace_place import ObjectGaussians,replace_fill,write_scene
        vid=view_id.strip()
        if vid not in cameras.cameras.by_id:
            raise ValueError(f"Missing camera {vid!r}")
        cam=cameras.cameras.by_id[vid]
        removed=_remove_scene(remove_scene_ply)
        source,ids,plane,remove_fill,rgb=_replace_scene(scene,cameras,object_ply,
                                                        "" if removed is not None else remove_fill_ply)
        output=_output_ply(output_path,"replace_base.ply")
        if removed is not None:
            total=write_scene(removed,np.ones(len(removed.vertex),bool),ObjectGaussians.empty(),output,
                              "SpEdE Replace base: the Remove's result")
            report={"base":"remove","remove_scene_ply":str(remove_scene_ply),"object_gaussians":int(len(ids))}
        else:
            drop,fill,report=replace_fill(source.means,source.opacity_logits,rgb,ids,plane,remove_fill=remove_fill)
            total=write_scene(source,~drop,fill,output,f"SpEdE Replace base: {len(ids)} object Gaussians out, "
                                                        f"{fill.count} fill Gaussians in")
            report["base"]="source"
        base=LocalScene(output,int(total),(source if removed is None else removed).max_sh_degree)
        if str(hero_image_path or "").strip():
            with Image.open(_input_path(hero_image_path)) as im:
                hero=np.asarray(im.convert("RGB").resize((cam.width,cam.height),Image.LANCZOS))
            report["hero_image"]=str(hero_image_path)
        else:
            hero,_alpha,_depth=_renderer(base,cameras,device).render_rgbd(vid,edited=False)
            report["hero_image"]="rendered base"
        report.update(scene_ply=str(output),total_gaussians=int(total),plane=plane.as_report())
        _write_report(base,"replace_base_report",report)
        return base,_tensor_image(hero),json.dumps(report)


class LocalPlaceGenerated:
    """Replace: the TripoSplat object stood where the removed one stood, in
    the source scene without exactly the extracted object ("inverse
    extract": the mat round it untouched).

    The support plane of the removed object, yaw from the silhouette and
    colour against Qwen's picture, base on the plane, position from where
    Qwen drew it (capped at ``max_shift`` of the old footprint's size), the
    scale by ``size`` (natural: Qwen's silhouette area; match: the removed
    object's dominant extent; larger: 1.3 x match), the part of the old
    contact footprint the new object leaves bare filled (from the Remove fill
    when ``remove_fill_ply`` is given), and a procedural contact shadow
    (spede_core.replace_place). With ``remove_scene_ply`` (a Remove's
    result) the object goes into that scene as it is: nothing taken out or
    filled, the source and the extract only measure the old object. Writes the scene PLY, the generated object as
    received and a ``.replace_report.json`` next to the scene."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "scene":(SCENE,),"cameras":(CAMERAS,),
            "view_id":("STRING",{"default":""}),
            "object_ply":("STRING",{"default":""}),
            "generated":("FILE_3D_SPLAT_ANY",),
            "inserted_image":("IMAGE",),"object_mask":("MASK",),
            "max_shift":("FLOAT",{"default":0.25,"min":0.0,"max":1.0,"step":0.05}),
            "size":(["match","natural","larger"],{"default":"match"}),
            "output_path":("STRING",{"default":"spede/local/replaced_scene.ply"}),
            "device":("STRING",{"default":"cuda"}),
        },"optional":{
            "remove_fill_ply":("STRING",{"default":""}),
            "remove_scene_ply":("STRING",{"default":""}),
        }}
    RETURN_TYPES=(SCENE,REPORT)
    RETURN_NAMES=("replaced_scene","replace_report_json")
    FUNCTION="place"
    CATEGORY="SpEdE Local/Replace"

    def place(self,scene,cameras,view_id,object_ply,generated,inserted_image,object_mask,
              max_shift,size,output_path,device,remove_fill_ply="",remove_scene_ply=""):
        import time
        from PIL import Image
        from spede_core.replace_place import (ObjectGaussians,contact_shadow,gsplat_render,keep_largest_component,
                                              load_object_ply,place_object,replace_fill,write_scene)
        started=time.time()
        views=cameras.cameras
        vid=view_id.strip()
        if vid not in views.by_id:
            raise ValueError(f"Missing camera {vid!r}")
        cam=views.by_id[vid]
        removed=_remove_scene(remove_scene_ply)
        source,ids,plane,remove_fill,rgb=_replace_scene(scene,cameras,object_ply,
                                                        "" if removed is not None else remove_fill_ply)
        output=_output_ply(output_path,"replaced_scene.ply")
        received=output.with_name(output.stem+".generated.ply")
        received.write_bytes(generated.get_bytes())
        obj,component=keep_largest_component(load_object_ply(received))
        # Qwen's picture and SAM's mask, in the hero camera's pixels.
        image=_array(inserted_image)[0,:,:,:3]
        mask=_array(object_mask)
        mask=mask.reshape(-1,*mask.shape[-2:]).max(axis=0)
        dims=(cam.width,cam.height)
        image=np.asarray(Image.fromarray(np.clip(image*255+0.5,0,255).astype(np.uint8)).resize(dims,Image.LANCZOS))
        mask=np.asarray(Image.fromarray((mask>0.5).astype(np.uint8)*255).resize(dims,Image.BILINEAR))>127
        Image.fromarray(mask.astype(np.uint8)*255).save(output.with_name(output.stem+".object_mask.png"))
        placed,placement=place_object(obj,plane=plane,viewmat=cam.viewmat,intrinsic=cam.intrinsic,mask=mask,
                                      image=image,max_shift_fraction=float(max_shift),size=str(size),
                                      render=None if device=="cpu" else gsplat_render(device))
        if removed is not None:
            # On the Remove's result: the old object is gone there and its spot filled.
            base,keep,fill,filled=removed,np.ones(len(removed.vertex),bool),ObjectGaussians.empty(),None
            message=f"SpEdE Replace on the Remove's result: {placed.count} generated"
        else:
            drop,fill,filled=replace_fill(source.means,source.opacity_logits,rgb,ids,plane,new_object=placed,
                                          remove_fill=remove_fill)
            base,keep=source,~drop
            message=f"SpEdE Replace: {len(ids)} object Gaussians out, {fill.count} fill, {placed.count} generated"
        # The shadow lies on what is there round the object: the base's surface and the fill.
        visible=keep&(base.opacity_logits.reshape(-1)>-0.85)               # opacity above 0.3
        surface=np.concatenate([base.means[visible].astype(np.float64),fill.means])
        shadow=contact_shadow(placed,plane,surface=surface,surface_band=0.05*plane.object_height)
        total=write_scene(base,keep,ObjectGaussians.concat(fill,placed,shadow),output,
                          f"{message}, {shadow.count} contact shadow")
        report={"status":"replaced","view_id":vid,"scene_ply":str(output),"source_scene":str(scene.ply),
                "base":"remove" if removed is not None else "source",
                "remove_scene_ply":str(remove_scene_ply) if removed is not None else None,
                "object_ply":str(object_ply),
                "remove_fill_ply":None if removed is not None else str(remove_fill_ply or "") or None,
                "generated_ply":str(received),"plane":plane.as_report(),"generated":component,
                "placement":placement,"fill":filled,"fill_gaussians":int(fill.count),
                "shadow_gaussians":int(shadow.count),"object_gaussians":int(placed.count),
                "total_gaussians":int(total),"seconds":round(time.time()-started,2)}
        _write_report(LocalScene(output,total,base.max_sh_degree),"replace_report",report)
        return LocalScene(output,int(total),base.max_sh_degree),json.dumps(report)

# ---- all-around fill (spede_core.multiview_fill) ---------------------------------
# Two stages that meet on disk: stage 1 (Qwen-Image chain, MoGe, the orbit
# warp) and stage 2 (MiniMax H3 orbit video, optimisation). In one prompt all
# their models would be held in RAM together (~56 GB).

ALLROUND="SPEDE_ALLROUND"
CHAIN_PROMPT=("Use <image1> as the authoritative camera image: edit <image1> only. Remove the {label} from <image1> "
              "completely, including its shadow and contact marks. Reconstruct only the surfaces hidden by it, "
              "preserving the camera viewpoint, framing, perspective, illumination and all other pixels of <image1>. "
              "Do not recreate the removed object. <image2> is only a reference: the same scene from a different "
              "camera with the {label} already removed. Do not copy its viewpoint; use it only to know what the "
              "hidden surface looks like (pattern, edges, colours). The result must have exactly the camera geometry "
              "and composition of <image1>.")
# The independent all-around fill's per-camera prompt: CHAIN_PROMPT without
# the reference image (each camera is cleaned on its own photo only).
INDEPENDENT_PROMPT=("Use <image1> as the authoritative camera image: edit <image1> only. Remove the {label} from "
                    "<image1> completely, including its shadow and contact marks. Reconstruct only the surfaces hidden "
                    "by it, preserving the camera viewpoint, framing, perspective, illumination and all other pixels "
                    "of <image1>. Do not recreate the removed object. The result must have exactly the camera "
                    "geometry and composition of <image1>.")


@dataclass(frozen=True)
class LocalAllRound:
    """The all-around fill's folder: state.json and every stage's results."""
    folder: Path

    @property
    def state(self)->dict:
        return json.loads((self.folder/"state.json").read_text())

    def update(self,**values)->"LocalAllRound":
        state={**self.state,**values}
        tmp=self.folder/"state.json.tmp"
        tmp.write_text(json.dumps(state,indent=2)+"\n")
        tmp.replace(self.folder/"state.json")
        return self

    def cameras(self):
        from spede_core.renderer import CameraSet
        return CameraSet.from_camera_pack(Path(self.state["camera_targets"]))

    def photos(self)->dict:
        with np.load(self.state["camera_targets"],allow_pickle=False) as f:
            return {str(l):f["targets"][i] for i,l in enumerate(f["labels"])}

    def masks(self)->dict:
        """The selection per camera: the hero's from its alignment, the
        others' SAM 3.1 masks."""
        from PIL import Image
        s=self.state
        out={}
        for v,row in json.loads(Path(s["segment_report"]).read_text())["views"].items():
            with Image.open(row["mask_path"]) as im:
                out[v]=np.asarray(im.convert("L"))>127
        out[s["hero"]]=np.load(Path(s["run_dir"])/"depth/alignment_inputs.npz")["mask"]
        return out

    def object_ids(self)->np.ndarray:
        return read_selected_ids(Path(self.state["run_dir"])/"selection/object_ids.npz",int(self.state["source_count"]))

    def chain_images(self)->dict:
        s=self.state
        return {v:np.asarray(_load_rgb(self.folder/"chain"/f"{k}_{v}_qwen.png"),np.float32)/255
                for k,v in enumerate(s["chain"])}


def _load_rgb(path,size=None):
    from PIL import Image
    with Image.open(path) as im:
        im=im.convert("RGB")
        if size is not None and im.size!=tuple(size):
            im=im.resize(tuple(size),Image.LANCZOS)
        return np.asarray(im)


def _save_rgb(array,path):
    from PIL import Image
    a=np.asarray(array)
    if a.dtype!=np.uint8:
        a=np.clip(a*255+0.5,0,255).astype(np.uint8)
    Image.fromarray(a[...,:3]).save(path)


def _sheet(tiles,columns=2,scale=0.5):
    """Rows of equal-size tiles as one picture."""
    from PIL import Image
    h,w=tiles[0].shape[:2]
    rows=(len(tiles)+columns-1)//columns
    sheet=np.zeros((rows*h,columns*w,3),np.uint8)
    for i,t in enumerate(tiles):
        sheet[(i//columns)*h:(i//columns+1)*h,(i%columns)*w:(i%columns+1)*w]=t[...,:3]
    im=Image.fromarray(sheet)
    return np.asarray(im.resize((int(im.width*scale),int(im.height*scale)),Image.LANCZOS))


def _free_comfy_vram():
    """The optimiser and the renders need the GPU the Comfy models hold."""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except ImportError:
        pass


def _hero_photo(state:dict,camera,size:int):
    """The hero photo at ``size`` px: the full-resolution capture photo's
    centre square (the camera pack's images are that square downsampled),
    verified against the pack's own image; without a photo folder the pack's
    image upsampled."""
    from PIL import Image
    small=np.asarray(state["_target"],np.float32)
    small=small/255 if small.max()>1.5 else small
    folder=str(state.get("photo_dir") or "").strip()
    if folder:
        hits=sorted(p for p in Path(folder).expanduser().glob(state["hero"]+".*")
                    if p.suffix.lower() in {".jpg",".jpeg",".png"})
        if not hits:
            raise FileNotFoundError(f"No photo {state['hero']}.* in {folder}")
        with Image.open(hits[0]) as im:
            im=im.convert("RGB")
            W,H=im.size
            s=min(W,H)
            crop=im.crop(((W-s)//2,(H-s)//2,(W-s)//2+s,(H-s)//2+s))
        check=np.asarray(crop.resize((camera.width,camera.height),Image.LANCZOS),np.float32)/255
        error=float(np.abs(check-small).mean())
        if error>0.06:
            raise ValueError(f"{hits[0]}'s centre square is not the camera's picture (mean error {error:.3f}); "
                             "leave photo_dir empty to use the camera pack's image")
        return np.asarray(crop.resize((size,size),Image.LANCZOS)),{"source":str(hits[0]),"check_error":round(error,4)}
    up=Image.fromarray((small*255).astype(np.uint8)).resize((size,size),Image.LANCZOS)
    return np.asarray(up),{"source":"camera pack image upsampled"}


class LocalAllRoundStart:
    """Plan the all-around fill from a finished depth-fill run: the chain of
    cameras for Qwen-Image, the orbit through real cameras, and the hero photo
    and removed scene at the orbit's resolution (for the hero's clean image)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "run_dir":("STRING",{"default":"","tooltip":"The depth-fill workflow's output folder (spede/local/<job>)."}),
            "source_ply":("STRING",{"default":"/PATH/TO/source_scene.ply"}),
            "camera_targets_npz":("STRING",{"default":"/PATH/TO/camera_pack.npz"}),
            "object_text":("STRING",{"default":"the object to remove"}),
            "output_subdir":("STRING",{"default":"spede/allround"}),
            "chain_views":("INT",{"default":10,"min":3,"max":24}),
            "orbit_frames":("INT",{"default":260,"min":124,"max":362,"step":17,
                                    "tooltip":"MiniMax H3 clips are 17k+5 frames; trained range 124-362."}),
            "resolution":("INT",{"default":1024,"min":512,"max":1536,"step":512}),
        },"optional":{
            "photo_dir":("STRING",{"default":"","tooltip":"Full-resolution capture photos (<view>.JPG); empty upsamples the camera pack."}),
            "device":("STRING",{"default":"cuda"}),
        }}
    RETURN_TYPES=(ALLROUND,"IMAGE","IMAGE",REPORT)
    RETURN_NAMES=("allround","hero_photo","removed_render","plan_json")
    FUNCTION="start"
    CATEGORY="SpEdE Local/All-around"

    def start(self,run_dir,source_ply,camera_targets_npz,object_text,output_subdir,chain_views,orbit_frames,
              resolution,photo_dir="",device="cuda"):
        from spede_core.multiview_fill import choose_chain,choose_orbit_stops,orbit_path
        from spede_core.renderer import CameraSet,CameraView,GsplatRenderer
        from spede_core.view_rank import rank_views
        if (int(orbit_frames)-5)%17:
            raise ValueError(f"orbit_frames must be 17k+5 (124, 141, ..., 260, ..., 362), not {orbit_frames}")
        run=_output_dir(run_dir,"local")
        # The selection's report (masks cut to the hero's instances) when
        # the run has one.
        segment=run/"selection/segment_report.json"
        if not segment.exists():
            segment=run/"masks/segment_report.json"
        for need in (segment,run/"selection/object_ids.npz",run/"depth/alignment_inputs.npz",
                     run/"depth/aligned_depth.npz",run/"depth_patch_scene.ply",run/"removed_scene.ply"):
            if not need.exists():
                raise FileNotFoundError(f"{need} missing: run_dir must be a finished depth-fill run")
        source=_input_path(source_ply)
        targets=_input_path(camera_targets_npz)
        info=json.loads(segment.read_text())
        hero=str(info["hero_view_id"])
        cams=CameraSet.from_camera_pack(targets)
        if cams.by_id[hero].width!=cams.by_id[hero].height or int(resolution)%cams.by_id[hero].width:
            raise ValueError("The all-around fill expects square camera images that divide the resolution")
        scene=GaussianPly.load(source)
        count=len(scene.means)
        ids=read_selected_ids(run/"selection/object_ids.npz",count)
        centre=np.median(scene.means[ids],axis=0)
        del scene
        # Scored for the surface the hero mask selects (view_rank, as the UI
        # ranks cameras): scoring the whole removed object (its underside and
        # footprint too) chose far views whose depth and carving misled the
        # fill (kitchen: 12 000 splats drifted against 3 400).
        renderer=GsplatRenderer(source_ply=source,cameras=cams,device=device)
        hero_mask=np.load(run/"depth/alignment_inputs.npz")["mask"]
        ranked=rank_views(renderer,cams,hero,hero_mask)
        del renderer
        if ranked["status"]!="ranked":
            raise RuntimeError(f"No camera sees the selection: {ranked['status']}")
        rows=ranked["views"]
        chain=choose_chain(rows,cams,hero,centre,count=int(chain_views),allowed=set(info["views"])|{hero})
        stops=choose_orbit_stops(rows,cams,hero,centre)
        path=orbit_path(cams,hero,stops,centre,int(orbit_frames))
        out=_output_dir(output_subdir,"allround")
        for sub in ("chain","warp","video","previews"):
            (out/sub).mkdir(exist_ok=True)
        scale=int(resolution)//cams.by_id[hero].width
        state={"run_dir":str(run),"source_ply":str(source),"camera_targets":str(targets),"segment_report":str(segment),
               "source_count":count,"hero":hero,"label":str(object_text).strip(),"centre":centre.tolist(),
               "chain":chain["chain"],"angle_to_hero":chain["angle_to_hero"],"orbit_stops":path["stops"],
               "orbit_keys":path["keys"],"frames":int(orbit_frames),"scale":scale,"resolution":int(resolution),
               "photo_dir":str(photo_dir or "")}
        (out/"orbit_path.json").write_text(json.dumps({"viewmats":path["viewmats"],
                                                       "intrinsic":np.asarray(cams.by_id[hero].intrinsic).tolist()}))
        (out/"view_scores.json").write_text(json.dumps([{k:v for k,v in r.items() if k!="points"} for r in rows],indent=1))
        (out/"state.json").write_text(json.dumps(state,indent=2)+"\n")
        cam=cams.by_id[hero]
        photo,photo_info=_hero_photo({**state,"_target":self._target(targets,hero)},cam,int(resolution))
        K=np.asarray(cam.intrinsic,np.float32).copy()
        K[:2]*=scale
        big=CameraView(view_id=f"{hero}@{resolution}",viewmat=np.asarray(cam.viewmat,np.float32),intrinsic=K,
                       width=int(resolution),height=int(resolution),synthetic=True)
        render=GsplatRenderer(source_ply=run/"removed_scene.ply",cameras=CameraSet([big]),device=device)\
            .render(big.view_id,edited=False)
        _save_rgb(photo,out/"hero_photo.png")
        _save_rgb(np.asarray(render),out/"hero_removed_render.png")
        report={**state,"hero_photo":photo_info,"views_scored":len(rows)}
        LocalAllRound(out).update(hero_photo=photo_info)
        return LocalAllRound(out),_tensor_image(photo),_tensor_image(np.asarray(render)),json.dumps(report)

    @staticmethod
    def _target(targets,view):
        with np.load(targets,allow_pickle=False) as f:
            labels=[str(l) for l in f["labels"]]
            return np.asarray(f["targets"][labels.index(view)])


class LocalAllRoundQwenChain:
    """Qwen-Image 2.1 cleans the chain cameras one by one: each gets its own
    photo and the nearest camera already cleaned (the hero's high-resolution
    clean image first) as the only reference: several references made Qwen
    copy a reference's composition. A result that moved more than 5 % of the
    picture outside the selection (reframed) is redone with the next seed."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "allround":(ALLROUND,),
            "model":("MODEL",),"clip":("CLIP",),"vae":("VAE",),
            "hero_clean":("IMAGE",{"tooltip":"Qwen-Image's clean hero at the orbit resolution."}),
            "seed":("INT",{"default":241007,"min":0,"max":2**63-1}),
            "steps":("INT",{"default":25,"min":1,"max":100}),
            "max_reframe":("FLOAT",{"default":0.05,"min":0.0,"max":1.0,"step":0.01}),
            "attempts":("INT",{"default":3,"min":1,"max":8}),
        }}
    RETURN_TYPES=(ALLROUND,"IMAGE")
    RETURN_NAMES=("allround","chain_sheet")
    FUNCTION="chain"
    CATEGORY="SpEdE Local/All-around"

    def chain(self,allround,model,clip,vae,hero_clean,seed,steps,max_reframe,attempts):
        import torch
        from PIL import Image,ImageDraw
        from scipy import ndimage
        import nodes as comfy_nodes                 # ComfyUI's own module, not this file
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        import comfy.utils
        s=allround.state
        chain,hero=s["chain"],s["hero"]
        cams=allround.cameras()
        photos=allround.photos()
        masks=allround.masks()
        folder=allround.folder/"chain"
        hero_big=_array(hero_clean)[0,:,:,:3]
        _save_rgb(hero_big,allround.folder/"hero_clean.png")
        size=(cams.by_id[hero].width,cams.by_id[hero].height)
        hero_small=np.asarray(Image.fromarray((hero_big*255+0.5).clip(0,255).astype(np.uint8)).resize(size,Image.LANCZOS))
        Image.fromarray(hero_small).save(folder/f"0_{hero}_qwen.png")

        def centre(v):
            return np.linalg.inv(np.asarray(cams.by_id[v].viewmat,np.float64))[:3,3]
        target=np.asarray(s["centre"])
        def direction(v):
            x=target-centre(v)
            return x/np.linalg.norm(x)

        def qwen(image1,image2,seed_):
            t1=torch.from_numpy(np.asarray(image1,np.float32)[None]/255)
            t2=torch.from_numpy(np.asarray(image2,np.float32)[None]/255)
            enc=TextEncodeQwenImage21.execute(clip,CHAIN_PROMPT.format(label=s["label"]),"",vae=vae,resolution=0,
                                              images={"image_1":t1,"image_2":t2}).args
            latent=comfy_nodes.common_ksampler(model,int(seed_),int(steps),1.0,"euler","simple",enc[0],enc[1],enc[2],
                                               denoise=1.0)[0]
            img=vae.decode(latent["samples"])
            if img.ndim==5:
                img=img.reshape(-1,*img.shape[-3:])
            return (img[0,...,:3].float().cpu().numpy()*255+0.5).clip(0,255).astype(np.uint8)

        moved={hero:0.0}
        done={hero:hero_small}
        bar=comfy.utils.ProgressBar(len(chain)-1)
        for k,v in enumerate(chain[1:],start=1):
            orig=photos[v]
            orig=(orig if orig.max()>1.5 else orig*255).astype(np.uint8)
            Image.fromarray(orig).save(folder/f"{k}_{v}_original.png")
            path=folder/f"{k}_{v}_qwen.png"
            near=max(done,key=lambda c:float(direction(c)@direction(v)))
            outside=~ndimage.binary_dilation(masks[v],iterations=30)
            if path.exists():                              # a resumed run keeps what it made
                result=_load_rgb(path)
            else:
                for attempt in range(int(attempts)):
                    result=qwen(orig,done[near],int(seed)+attempt)
                    if result.shape[:2]!=orig.shape[:2]:
                        result=np.asarray(Image.fromarray(result).resize(orig.shape[1::-1],Image.LANCZOS))
                    diff=ndimage.median_filter(np.abs(orig.astype(np.float32)-result).mean(2)/255,5)
                    moved[v]=float((diff[outside]>0.15).mean())
                    if moved[v]<float(max_reframe):
                        break
                Image.fromarray(result).save(path)
            if v not in moved:
                diff=ndimage.median_filter(np.abs(orig.astype(np.float32)-result).mean(2)/255,5)
                moved[v]=float((diff[outside]>0.15).mean())
            done[v]=result
            bar.update(1)
        tiles=[]
        for k,v in enumerate(chain):
            o=photos[v]
            o=(o if o.max()>1.5 else o*255).astype(np.uint8).copy()
            m=masks[v]
            o[m^ndimage.binary_erosion(m,iterations=2)]=[0,255,255]
            row=Image.fromarray(np.concatenate([o,np.asarray(done[v])],1))
            ImageDraw.Draw(row).text((8,8),f"{v}  {s['angle_to_hero'][v]} deg from hero, outside changed "
                                           f"{moved[v]:.0%}",fill=(255,255,0))
            tiles.append(np.asarray(row))
        sheet=_sheet(tiles)
        _save_rgb(sheet,allround.folder/"chain_sheet.png")
        reframed=[v for v,x in moved.items() if x>=float(max_reframe)]
        allround.update(chain_outside_changed={v:round(x,3) for v,x in moved.items()},chain_reframed=reframed)
        return allround,_tensor_image(sheet)


class LocalAllRoundInit:
    """Every chain camera's clean image adds depth births to the hero's
    depth patch where it disagrees with the current render (its MoGe 3 depth
    aligned to the scene; residue and shadow carved as the hero does)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"allround":(ALLROUND,),"moge_model":("MOGE_MODEL",)},
                "optional":{"device":("STRING",{"default":"cuda"})}}
    RETURN_TYPES=(ALLROUND,SCENE,"IMAGE")
    RETURN_NAMES=("allround","init_scene","chain_renders")
    FUNCTION="init"
    CATEGORY="SpEdE Local/All-around"

    def init(self,allround,moge_model,device="cuda"):
        import torch
        from scipy import ndimage
        from spede_core.multiview_fill import multiview_init
        from spede_core.renderer import GsplatRenderer
        s=allround.state
        cams=allround.cameras()
        masks=allround.masks()
        chain=s["chain"]
        grown={v:(masks[v] if v==s["hero"] else ndimage.binary_dilation(masks[v],iterations=3)) for v in chain}
        edits=allround.chain_images()

        def moge_depth(rgb,cam):
            fov=float(np.degrees(2*np.arctan(cam.width/(2*float(cam.intrinsic[0][0])))))
            t=torch.from_numpy(np.asarray(rgb,np.float32)).permute(2,0,1)[None]
            with torch.no_grad():
                r=moge_model.infer(t,resolution_level=7,fov_x=fov,force_projection=True,apply_mask=False,refine_steps=0)
            valid=(r["mask"][0].cpu().numpy()>0.5) if "mask" in r else None
            return r["depth"][0].float().cpu().numpy(),valid

        lines=[]
        def log(text):
            lines.append(text)
            LOG.info("[SpEdE all-around init] %s",text)
        _free_comfy_vram()
        result=multiview_init(source_ply=Path(s["source_ply"]),hero_patch_ply=Path(s["run_dir"])/"depth_patch_scene.ply",
                              cameras=cams,chain=chain,masks=grown,edits=edits,object_ids=allround.object_ids(),
                              moge_depth=moge_depth,out_dir=allround.folder/"init",device=device,log=log)
        (allround.folder/"init/init_report.json").write_text(json.dumps(result,indent=2,default=str))
        tiles=[]
        r=GsplatRenderer(source_ply=Path(result["ply"]),cameras=cams,device=device)
        for v in chain:
            img=np.asarray(r.render(v,edited=False))
            tiles.append(np.concatenate([(edits[v]*255).astype(np.uint8),img[...,:3]],1))
        del r
        sheet=_sheet(tiles)
        _save_rgb(sheet,allround.folder/"init_sheet.png")
        allround.update(init_ply=result["ply"],init_steps=result["steps"])
        count=len(GaussianPly.load(Path(result["ply"])).means)
        return allround,LocalScene(Path(result["ply"]),count,GaussianPly.load(Path(s["source_ply"])).max_sh_degree),_tensor_image(sheet)


def _allround_outputs(allround):
    """The orbit warp and the hero's clean image, as MiniMax H3 takes them."""
    import torch
    s=allround.state
    frames=sorted((allround.folder/"warp").glob("warp_*.png"))
    if len(frames)!=s["frames"]:
        raise FileNotFoundError(f"{allround.folder/'warp'} has {len(frames)} of {s['frames']} warp frames")
    batch=torch.from_numpy(np.stack([_load_rgb(p) for p in frames]).astype(np.float32)/255)
    hero=_tensor_image(_load_rgb(allround.folder/"hero_clean.png"))
    return allround,batch,hero,int(s["frames"]),int(s["resolution"])


class LocalAllRoundWarp:
    """CrossView-Warp conditioning on the orbit: the hero's clean image
    reprojected with its aligned depth into every orbit frame's camera
    (magenta where the hero saw nothing, as the LoRA was trained)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"allround":(ALLROUND,)}}
    RETURN_TYPES=(ALLROUND,"IMAGE","IMAGE","INT","INT")
    RETURN_NAMES=("allround","warp_video","hero_clean","frames","resolution")
    FUNCTION="warp"
    CATEGORY="SpEdE Local/All-around"

    def warp(self,allround):
        from PIL import Image
        from spede_core.multiview_fill import HeroWarp,hero_depth
        import comfy.utils
        s=allround.state
        cams=allround.cameras()
        hero=cams.by_id[s["hero"]]
        scale=int(s["scale"])
        run=Path(s["run_dir"])
        ai=np.load(run/"depth/alignment_inputs.npz")
        aligned=np.load(run/"depth/aligned_depth.npz")["camera_z"]
        z=hero_depth(ai["mask"],ai["source_depth"],ai["source_alpha"],aligned,scale)
        rgb=_load_rgb(allround.folder/"hero_clean.png",(z.shape[1],z.shape[0]))
        K=np.asarray(hero.intrinsic,np.float64).copy()
        K[:2]*=scale
        warp=HeroWarp(rgb,z,K,hero.viewmat)
        path=json.loads((allround.folder/"orbit_path.json").read_text())
        bar=comfy.utils.ProgressBar(len(path["viewmats"]))
        keys=set(int(round(k)) for k in s["orbit_keys"])
        tiles=[]
        for f,vm in enumerate(path["viewmats"]):
            img=warp.render(np.asarray(vm),K)
            Image.fromarray(img).save(allround.folder/"warp"/f"warp_{f:04d}.png")
            if f in keys:
                tiles.append(img)
            bar.update(1)
        _save_rgb(_sheet(tiles,columns=len(tiles),scale=0.25),allround.folder/"warp_sheet.png")
        allround.update(warp_done=True)
        return _allround_outputs(allround)


class LocalAllRoundHero:
    """The lean stage 1's end: the hero's 1024 clean image saved with the
    plan (``hero_clean.png``). The progressive and independent fills need
    nothing else of stage 1; the Qwen chain, MoGe births and orbit warp
    feed only the orbit path."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"allround":(ALLROUND,),
                            "hero_clean":("IMAGE",{"tooltip":"Qwen-Image's clean hero at the orbit resolution."})}}
    RETURN_TYPES=(ALLROUND,"IMAGE")
    RETURN_NAMES=("allround","hero_clean")
    FUNCTION="save"
    CATEGORY="SpEdE Local/All-around"

    def save(self,allround,hero_clean):
        hero=_array(hero_clean)[0,:,:,:3]
        _save_rgb(hero,allround.folder/"hero_clean.png")
        return allround,_tensor_image((hero*255+0.5).clip(0,255).astype(np.uint8))


class LocalAllRoundLoad:
    """A finished stage 1 folder. ``require`` warp (stage 2): the full stage 1
    (state.json, chain, warp); hero_clean (the progressive and independent
    fills): the plan and the hero's clean image, from either stage 1; the
    warp outputs are then the hero clean as one frame and 0 frames."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{"allround_dir":("STRING",{"default":"spede/allround"})},
                "optional":{"require":(["warp","hero_clean"],{"default":"warp"})}}
    RETURN_TYPES=(ALLROUND,"IMAGE","IMAGE","INT","INT")
    RETURN_NAMES=("allround","warp_video","hero_clean","frames","resolution")
    FUNCTION="load"
    CATEGORY="SpEdE Local/All-around"

    def load(self,allround_dir,require="warp"):
        folder=_output_dir(allround_dir,"allround")
        if not (folder/"state.json").exists():
            raise FileNotFoundError(f"{folder} is not an all-around stage 1 folder")
        allround=LocalAllRound(folder)
        if require=="hero_clean":
            if not (folder/"hero_clean.png").exists():
                raise FileNotFoundError(f"{folder} has no hero_clean.png: run stage 1 first")
            if allround.state.get("warp_done"):
                return _allround_outputs(allround)
            hero=_tensor_image(_load_rgb(folder/"hero_clean.png"))
            return allround,hero,hero,0,int(allround.state["resolution"])
        if not allround.state.get("warp_done"):
            raise FileNotFoundError(f"{folder} is not a finished all-around stage 1")
        return _allround_outputs(allround)


class LocalAllRoundOptimize:
    """The hole's Gaussians optimised against every camera at once: the
    chain's clean images and the orbit video's frames inside the hole (frames
    that slipped off their path dropped), every real photo outside it."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":{
            "allround":(ALLROUND,),
            "iterations":("INT",{"default":10000,"min":100,"max":100000}),
            "anchor":("BOOLEAN",{"default":True,"tooltip":"Keep the fill near where it started (no floating fuzz)."}),
            "freeze_view_colour":("BOOLEAN",{"default":True,"tooltip":"Do not train SH rest terms (haze in web viewers)."}),
            "output_path":("STRING",{"default":"spede/allround/allround_scene.ply"}),
        },"optional":{
            # Without it, the frames an earlier run saved (video/frame_*.png):
            # a failed optimisation resumes without filming the orbit again.
            "orbit_video":("IMAGE",{"tooltip":"MiniMax H3's CrossView-Warp video of the orbit."}),
            "device":("STRING",{"default":"cuda"})}}
    RETURN_TYPES=(SCENE,REPORT,"IMAGE")
    RETURN_NAMES=("scene","report_json","chain_renders")
    FUNCTION="optimize"
    CATEGORY="SpEdE Local/All-around"

    def optimize(self,allround,iterations,anchor,freeze_view_colour,output_path,orbit_video=None,device="cuda"):
        from PIL import Image
        from spede_core.multiview_fill import frame_check,multiview_optimize,object_hole
        import comfy.utils
        s=allround.state
        cams=allround.cameras()
        hero=cams.by_id[s["hero"]]
        size=hero.width
        path=json.loads((allround.folder/"orbit_path.json").read_text())
        if orbit_video is not None:
            video=_array(orbit_video)[...,:3]
        else:
            saved=sorted((allround.folder/"video").glob("frame_*.png"))
            if not saved:
                raise FileNotFoundError(f"No orbit_video and no saved frames in {allround.folder/'video'}")
            video=np.stack([_load_rgb(p) for p in saved]).astype(np.float32)/255
        n=min(len(video),len(path["viewmats"]))
        if n<len(path["viewmats"])*0.8:
            raise ValueError(f"The video has {len(video)} frames for an orbit of {len(path['viewmats'])}")
        obj=GaussianPly.load(Path(s["source_ply"])).means[allround.object_ids()]
        K=np.asarray(path["intrinsic"],np.float32)
        frames,checks=[],[]
        for f in range(n):
            frame=(video[f]*255+0.5).clip(0,255).astype(np.uint8)
            if orbit_video is not None:
                Image.fromarray(frame).save(allround.folder/"video"/f"frame_{f:04d}.png")
            check=frame_check(_load_rgb(allround.folder/"warp"/f"warp_{f:04d}.png"),frame,warp_scale=int(s["scale"]))
            checks.append({"f":f,**check})
            if check["usable"]:
                small=np.asarray(Image.fromarray(frame).resize((size,size),Image.LANCZOS),np.float32)/255
                frames.append({"frame":f,"viewmat":path["viewmats"][f],"intrinsic":K,"image":small,
                               "hole":object_hole(obj,path["viewmats"][f],K,size)})
        (allround.folder/"video/check.json").write_text(json.dumps({"usable":[c["f"] for c in checks if c["usable"]],
                                                                    "frames":checks},indent=1))
        if len(frames)<n*0.5:
            raise RuntimeError(f"Only {len(frames)} of {n} video frames stayed on their camera path: "
                               "H3 did not follow the warp; the fill would learn the wrong geometry")
        lines=[]
        def log(text):
            lines.append(text)
            LOG.info("[SpEdE all-around] %s",text)
        bar=comfy.utils.ProgressBar(int(iterations))
        _free_comfy_vram()
        out=_output_ply(output_path,"allround_scene.ply")
        masks=allround.masks()
        report=multiview_optimize(source_ply=Path(s["source_ply"]),init_ply=Path(s["init_ply"]),cameras=cams,
                                  photos=allround.photos(),masks=masks,chain=s["chain"],edits=allround.chain_images(),
                                  object_ids=allround.object_ids(),video=frames,out_ply=out,
                                  iterations=int(iterations),anchor=bool(anchor),
                                  freeze_view_colour=bool(freeze_view_colour),device=device,log=log,
                                  progress=lambda i,_n:bar.update_absolute(i),preview_dir=allround.folder/"previews")
        report["usable_frames"]=len(frames)
        report["orbit_frames"]=n
        report["log"]=lines
        (out.with_suffix(".allround_report.json")).write_text(json.dumps(report,indent=2))
        allround.update(result_ply=str(out),report=str(out.with_suffix(".allround_report.json")))
        tiles=[]
        for v in s["chain"]:
            p=allround.folder/"previews"/f"after_{v}.png"
            if p.exists():
                tiles.append(np.concatenate([(allround.chain_images()[v]*255).astype(np.uint8),_load_rgb(p)],1))
        sheet=_sheet(tiles) if tiles else np.zeros((64,64,3),np.uint8)
        _save_rgb(sheet,allround.folder/"result_sheet.png")
        sh=GaussianPly.load(Path(s["source_ply"])).max_sh_degree
        return LocalScene(out,int(report["total_gaussians"]),sh),json.dumps({k:v for k,v in report.items() if k!="log"}),_tensor_image(sheet)

TURBO_SIGMAS="1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0"


class LocalAllRoundProgressive:
    """The all-around fill (spede_core/progressive_fill.py). ``method``
    independent (the default): Qwen-Image cleans every camera that sees the
    hole on its own photo, with no reference, then one optimisation against
    all of them (``independent_fill``). progressive: one camera at a time,
    each seeing the fill so far: the current scene rendered into the next
    camera, Qwen-Image cleans only the hole (uncovered pixels marked magenta,
    the nearest finished camera as reference), births only where the render
    was empty, then a short optimisation against every camera done
    (``progressive_fill``; it alone uses MoGe, which is loaded only then).
    Both build the support surface and paint it from above with ``prompt``."""
    @classmethod
    def INPUT_TYPES(cls):
        from spede_core.progressive_fill import CLEAN_PROMPT
        return {"required":{
            "allround":(ALLROUND,),
            "model":("MODEL",{"tooltip":"Qwen-Image 2.1 (turbo: 8 steps with the sigmas below)."}),
            "clip":("CLIP",),"vae":("VAE",),
            "moge_model":("MOGE_MODEL",{"lazy":True,"tooltip":"MoGe-3: the progressive method's births only."}),
            "hero_clean":("IMAGE",{"tooltip":"The hero's clean image (Qwen-Image)."}),
            "seed":("INT",{"default":241007,"min":0,"max":2**63-1}),
            "sigmas":("STRING",{"default":TURBO_SIGMAS}),
            "prompt":("STRING",{"default":CLEAN_PROMPT,"multiline":True}),
            "step_iterations":("INT",{"default":1500,"min":0,"max":20000}),
            "final_iterations":("INT",{"default":6000,"min":0,"max":100000}),
            "output_path":("STRING",{"default":"spede/allround/progressive/progressive_scene.ply"}),
        },"optional":{"device":("STRING",{"default":"cuda"}),
                      # After device: saved UI workflows fill widgets in order.
                      "method":(["independent","progressive"],{"default":"independent",
                                "tooltip":"independent: each camera cleaned on its own photo, one optimisation; "
                                          "progressive: camera by camera with births (slower)."})}}
    RETURN_TYPES=("SPEDE_SCENE","STRING","IMAGE")
    RETURN_NAMES=("scene","report_json","sheet")
    FUNCTION="run"
    CATEGORY="SpEdE Local/All-around"

    def check_lazy_status(self,method="independent",moge_model=None,**_):
        return ["moge_model"] if method=="progressive" and moge_model is None else []

    def run(self,allround,model,clip,vae,hero_clean,seed,sigmas,prompt,step_iterations,final_iterations,
            output_path,device="cuda",method="independent",moge_model=None):
        import torch
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21
        from comfy_extras.nodes_custom_sampler import (CFGGuider,KSamplerSelect,ManualSigmas,RandomNoise,
                                                       SamplerCustomAdvanced)
        import comfy.utils
        import spede_core.progressive_fill as pf
        s=allround.state
        cams=allround.cameras()
        sig=ManualSigmas.execute(sigmas).args[0]
        sampler=KSamplerSelect.execute("euler").args[0]

        def edit(text,images,resolution,seed_):
            enc=TextEncodeQwenImage21.execute(clip,text,"",vae=vae,resolution=int(resolution),images=images).args
            guider=CFGGuider.execute(model,enc[0],enc[1],1.0).args[0]
            noise=RandomNoise.execute(int(seed_)).args[0]
            latent=SamplerCustomAdvanced.execute(noise,guider,sampler,sig,enc[2]).args[0]
            img=vae.decode(latent["samples"])
            if img.ndim==5:
                img=img.reshape(-1,*img.shape[-3:])
            return (img[0,...,:3].float().cpu().numpy()*255+0.5).clip(0,255).astype(np.uint8)

        def tensor(a):
            return torch.from_numpy(np.asarray(a,np.float32)[None]/255)

        def clean(marked,reference,seed_):
            return edit(prompt,{"image_1":tensor(marked),"image_2":tensor(reference)},marked.shape[0],seed_)

        single=INDEPENDENT_PROMPT.format(label=s["label"])

        def clean_single(image,seed_):
            return edit(single,{"image_1":tensor(image)},image.shape[0],seed_)

        def moge_depth(rgb,cam):
            fov=float(np.degrees(2*np.arctan(cam.width/(2*float(cam.intrinsic[0][0])))))
            t=torch.from_numpy(np.asarray(rgb,np.float32)).permute(2,0,1)[None]
            with torch.no_grad():
                r=moge_model.infer(t,resolution_level=7,fov_x=fov,force_projection=True,apply_mask=False,refine_steps=0)
            valid=(r["mask"][0].cpu().numpy()>0.5) if "mask" in r else None
            return r["depth"][0].float().cpu().numpy(),valid

        lines=[]
        def log(text):
            lines.append(text)
            LOG.info("[SpEdE progressive] %s",text)
        out=_output_ply(output_path,"progressive_scene.ply")
        folder=out.parent
        hero=_array(hero_clean)[0,:,:,:3]
        bar=comfy.utils.ProgressBar(100)
        photos=allround.photos()
        common=dict(source_ply=Path(s["source_ply"]),start_ply=Path(s["run_dir"])/"depth_patch_scene.ply",cameras=cams,
                    hero=s["hero"],hero_clean=(hero*255+0.5).clip(0,255).astype(np.uint8),photos=photos,
                    selections=allround.masks(),object_ids=allround.object_ids(),centre=np.asarray(s["centre"]),
                    clean=clean,out_dir=folder,seed=int(seed),final_iterations=int(final_iterations),device=device,
                    log=log,progress=lambda i,n:bar.update_absolute(int(100*i/max(n,1)),100))
        if method=="progressive":
            report=pf.progressive_fill(moge_depth=moge_depth,step_iterations=int(step_iterations),**common)
        else:
            def camera_image(v):
                # The full-resolution photo's centre square, as the hero's.
                try:
                    return _hero_photo({**s,"hero":v,"_target":photos[v]},cams.by_id[v],1024)[0]
                except (FileNotFoundError,ValueError) as e:
                    log(f"{v}: {e}; the camera pack's image upsampled instead")
                    return _hero_photo({**s,"hero":v,"photo_dir":"","_target":photos[v]},cams.by_id[v],1024)[0]
            report=pf.independent_fill(clean_single=clean_single,camera_image=camera_image,**common)
        if Path(report["ply"])!=out:
            shutil.move(report["ply"],out)
        report["ply"]=str(out)
        report["log"]=lines
        out.with_suffix(".progressive_report.json").write_text(json.dumps(report,indent=1,default=str))
        tiles=[]
        for v in report["order"]:
            c=sorted((folder/"views").glob(f"*_{v}_clean.png"))
            p=folder/"previews"/f"after_{v}.png"
            if c and p.exists():
                tiles.append(np.concatenate([_load_rgb(c[0],_load_rgb(p).shape[1::-1]),_load_rgb(p)],1))
        sheet=_sheet(tiles) if tiles else np.zeros((64,64,3),np.uint8)
        _save_rgb(sheet,folder/"progressive_sheet.png")
        sh=GaussianPly.load(Path(s["source_ply"])).max_sh_degree
        count=len(GaussianPly.load(out).means)
        return LocalScene(out,count,sh),json.dumps({k:v for k,v in report.items() if k not in ("log","steps")},default=str),_tensor_image(sheet)

NODE_CLASS_MAPPINGS={
    "SpEdE_LocalAllRoundStart":LocalAllRoundStart,
    "SpEdE_LocalAllRoundQwenChain":LocalAllRoundQwenChain,
    "SpEdE_LocalAllRoundInit":LocalAllRoundInit,
    "SpEdE_LocalAllRoundWarp":LocalAllRoundWarp,
    "SpEdE_LocalAllRoundHero":LocalAllRoundHero,
    "SpEdE_LocalAllRoundLoad":LocalAllRoundLoad,
    "SpEdE_LocalAllRoundOptimize":LocalAllRoundOptimize,
    "SpEdE_LocalAllRoundProgressive":LocalAllRoundProgressive,
    "SpEdE_LocalPrepareReplace":LocalPrepareReplace,
    "SpEdE_LocalRegisterSplat":LocalRegisterSplat,
    "SpEdE_LocalObjectCutout":LocalObjectCutout,
    "SpEdE_LocalReplaceBase":LocalReplaceBase,
    "SpEdE_LocalPlaceGenerated":LocalPlaceGenerated,
    "SpEdE_LocalCameraFOV":LocalCameraFOV,
    "SpEdE_LocalAlignMoGeDepth":LocalAlignMoGeDepth,
    "SpEdE_LocalPatchFromDepth":LocalPatchFromDepth,
    "SpEdE_LocalViewImage":LocalViewImage,
    "SpEdE_LocalFocusCrop":LocalFocusCrop,
    "SpEdE_LocalObjectName":LocalObjectName,
    "SpEdE_LocalHeroConsistency":LocalHeroConsistency,
    "SpEdE_LocalHeldoutQA":LocalHeldoutQA,
    "SpEdE_LocalLoadScene":LocalLoadScene,
    "SpEdE_LocalLoadCameras":LocalLoadCameras,
    "SpEdE_LocalCameraImages":LocalCameraImages,
    "SpEdE_LocalMasksToReport":LocalMasksToReport,
    "SpEdE_LocalSelectMultiView":LocalSelectMultiView,
    "SpEdE_LocalExtract":LocalExtract,
    "SpEdE_LocalRemove":LocalRemove,
    "SpEdE_LocalMaskView":LocalMaskView,
    "SpEdE_LocalRender":LocalRender,
    "SpEdE_LocalPatchPlane":LocalPatchPlane,
    "SpEdE_LocalPatchQA":LocalPatchQA,
    "SpEdE_LocalSaveScene":LocalSaveScene,
}
NODE_DISPLAY_NAME_MAPPINGS={
    k:"SpEdE · "+k.removeprefix("SpEdE_Local") for k in NODE_CLASS_MAPPINGS
}
