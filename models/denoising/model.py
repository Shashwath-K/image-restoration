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
                 weights_path: Optional[str] = None,
                 use_fallback: bool = True):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_path = weights_path
        self.use_fallback = use_fallback

        self.model = None
        self.model_loaded = False

        if weights_path and os.path.isfile(weights_path):
            try:
                self._load_weights(weights_path)
                self.model_loaded = True
            except Exception as e:
                print(f"[DenoisingModel] Failed to load weights: {e}")
                self.model_loaded = False

    # ----------------------------------------------------------
    # Model loading (placeholder for DnCNN integration)
    # ----------------------------------------------------------
    def _load_weights(self, path: str):
        """
        Placeholder loader.
        Replace with actual DnCNN model initialization.
        """
        state = torch.load(path, map_location=self.device)
        # Example:
        # self.model = DnCNN(...)
        # self.model.load_state_dict(state)
        # self.model.to(self.device).eval()
        print(f"[DenoisingModel] (placeholder) Loaded weights from {path}")

    # ----------------------------------------------------------
    # Fallback: simple Gaussian smoothing
    # ----------------------------------------------------------
    @staticmethod
    def _gaussian_denoise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
        """
        Basic denoising using Gaussian blur.
        """
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Denoise an image.

        image : BGR uint8 numpy array
        returns: denoised image (same format)
        """
        if self.model_loaded and self.model is not None:
            # Real deep-learning inference goes here
            # Example pseudocode:
            # inp = preprocess(image).to(self.device)
            # with torch.no_grad():
            #     out = self.model(inp)
            # return postprocess(out)
            raise NotImplementedError("Deep model inference not wired yet.")
        else:
            if not self.use_fallback:
                raise RuntimeError("No model loaded and fallback disabled.")
            return self._gaussian_denoise(image)
