"""
services/inpaint_service.py

Service wrapper for image inpainting.
"""

from typing import Optional
import numpy as np
from .base import get_device
from models.inpainting import InpaintingModel

class InpaintService:
    def __init__(self,
                 device: Optional[str] = None,
                 weights_path: Optional[str] = None):
        self.device = get_device(device)
        
        if weights_path is None:
            weights_path = "models/inpainting/weights/inpainting.pth"

        self.model = InpaintingModel(
            device=self.device,
            weights_path=weights_path
        )

    def restore(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint missing regions in an image.

        image : BGR uint8 image
        mask  : uint8 mask (255 = missing region)
        """
        return self.model.inpaint(image, mask)
