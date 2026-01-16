"""
services/denoise_service.py

Service wrapper for image denoising.
"""

from typing import Optional
import numpy as np
from .base import get_device
from models.denoising import DenoisingModel

class DenoiseService:
    def __init__(self,
                 device: Optional[str] = None,
                 weights_path: Optional[str] = None):
        self.device = get_device(device)
        self.model = DenoisingModel(
            device=self.device,
            weights_path=weights_path,
            use_fallback=True
        )

    def restore(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise an image.

        image : BGR uint8 numpy array
        returns: denoised image
        """
        return self.model.denoise(image)
