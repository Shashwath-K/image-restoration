"""
services/colorize_service.py

Service wrapper for image colorization.
"""

from typing import Optional
import numpy as np
from .base import get_device
from models.colorization import ColorizationModel

class ColorizeService:
    def __init__(self,
                 device: Optional[str] = None,
                 weights_path: Optional[str] = None):
        self.device = get_device(device)

        if weights_path is None:
            weights_path = "models/colorization/weights/colorizer.pth"
            
        self.model = ColorizationModel(
            device=self.device,
            weights_path=weights_path
        )

    def restore(self, image: np.ndarray) -> np.ndarray:
        """
        Colorize a grayscale or low-color image.

        image : BGR uint8 image
        returns: BGR uint8 color image
        """
        return self.model.colorize(image)
