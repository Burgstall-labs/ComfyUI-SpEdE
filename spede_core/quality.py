"""Novel/capture-view spatial locality metrics for patched Gaussian scenes."""
from __future__ import annotations
import numpy as np
from scipy import ndimage


def outside_mask_preservation(source_rgb,patched_rgb,object_mask,
                              *, boundary_px:int=3, pixel_delta:float=0.12,
                              maximum_mean_abs:float=0.03,
                              maximum_changed_fraction:float=0.07)->dict:
    a=np.asarray(source_rgb,np.float32)
    b=np.asarray(patched_rgb,np.float32)
    mask=np.asarray(object_mask,bool)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[2]<3 or a.shape[:2]!=mask.shape:
        raise ValueError("Image/locality mask dimensions mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite RGB values in QA")
    if a.max()>1: a=a/255.0
    if b.max()>1: b=b/255.0
    excluded=ndimage.binary_dilation(mask,iterations=int(boundary_px))
    outside=~excluded
    if not outside.any():
        return {
            "passed":False, "reason":"no pixels outside object mask",
            "judged_pixels":0,
        }
    delta=np.abs(a[:,:,:3]-b[:,:,:3]).mean(axis=-1)
    average=float(delta[outside].mean())
    changed=float((delta[outside]>float(pixel_delta)).mean())
    return {
        "name":"H1 heldout outside-mask preservation",
        "passed":average<=maximum_mean_abs and changed<=maximum_changed_fraction,
        "outside_rgb_mae":average,
        "outside_changed_fraction":changed,
        "judged_pixels":int(outside.sum()),
        "maximum_mean_abs":float(maximum_mean_abs),
        "maximum_changed_fraction":float(maximum_changed_fraction),
    }
