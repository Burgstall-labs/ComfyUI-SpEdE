from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CandidateView:
    view_id: str
    camera_to_world: np.ndarray
    intrinsics: np.ndarray
    visible_gaussian_ids: np.ndarray
    projected_area: float
    occlusion_fraction: float
    angular_signature: np.ndarray
