"""
models/denoising/model.py

Wrapper for image denoising models.
Primary target: DnCNN.
Fallback: OpenCV Gaussian smoothing.
"""

from typing import Optional
import os
import cv2
import numpy as np
import torch

class DenoisingModel:
    def __init__(self,
                 device: Optional[str] = None,
                 weights_path: Optional[str] = None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path

        self.model = None
        self.model_loaded = False

        if weights_path and os.path.isfile(weights_path):
            try:
                self._load_weights(weights_path)
                self.model_loaded = True
            except Exception as e:
                print(f"[DenoisingModel] Failed to load weights: {e}")
                self.model_loaded = False
        else:
             print(f"[DenoisingModel] Weights not found at {weights_path}. Model will not be loaded.")

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_weights(self, path: str):
        """
        Load DnCNN model weights.
        """
        from .dncnn import DnCNN
        self.model = DnCNN(depth=17, n_channels=64, image_channels=3)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[DenoisingModel] Loaded weights from {path}")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise an image using DnCNN.

        image : BGR uint8 numpy array
        returns: denoised image (same format)
        """
        if self.model_loaded and self.model is not None:
             # Preprocess
            img_normalized = image.astype(np.float32) / 255.0
            # HWC -> CHW
            img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float().unsqueeze(0)
            img_tensor = img_tensor.to(self.device)

            with torch.no_grad():
                out_tensor = self.model(img_tensor)

            # Postprocess
            out_tensor = out_tensor.squeeze(0).cpu()
            out_img = out_tensor.numpy().transpose(1, 2, 0)
            out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
            return out_img
        else:
             raise RuntimeError("DnCNN model not loaded. Cannot perform denoising.")
