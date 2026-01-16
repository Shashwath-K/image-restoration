"""
services/interpolate_service.py

Service wrapper for frame interpolation.
"""

from typing import Optional, List
import numpy as np
from .base import get_device
from models.interpolation import InterpolationModel

class InterpolateService:
    def __init__(self,
                 device: Optional[str] = None,
                 weights_path: Optional[str] = None):
        self.device = get_device(device)
        self.model = InterpolationModel(
            device=self.device,
            weights_path=weights_path,
            fallback_blend=True
        )

    def restore(self,
                frames: List[np.ndarray],
                num_mid_frames: int = 1) -> List[np.ndarray]:
        """
        Interpolate missing frames in a sequence.

        frames : list of BGR uint8 frames
        returns: list of frames with intermediates inserted
        """
        return self.model.interpolate_sequence(frames, num_mid=num_mid_frames)
