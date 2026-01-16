"""
models/inpainting/model.py

Wrapper for deep-learning-based image inpainting.
Primary target: LaMa.
Fallback: OpenCV inpainting.
"""

from typing import Optional
import os
import cv2
import numpy as np
import torch

class InpaintingModel:
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
                print(f"[InpaintingModel] Failed to load weights: {e}")
                self.model_loaded = False

    # ------------------------------------------------------------------
    # Model loading (placeholder for LaMa integration)
    # ------------------------------------------------------------------
    def _load_weights(self, path: str):
        """
        Placeholder loader.
        Replace with LaMa model initialization logic.
        """
        state = torch.load(path, map_location=self.device)
        # Example:
        # self.model = LamaModel(...)
        # self.model.load_state_dict(state)
        # self.model.to(self.device).eval()
        print(f"[InpaintingModel] (placeholder) Loaded weights from {path}")

    # ------------------------------------------------------------------
    # Fallback: OpenCV inpainting
    # ------------------------------------------------------------------
    @staticmethod
    def _opencv_inpaint(image: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
        """
        image : BGR uint8 image
        mask  : uint8 mask (255 = missing region)
        """
        if method.lower() == "ns":
            flag = cv2.INPAINT_NS
        else:
            flag = cv2.INPAINT_TELEA

        return cv2.inpaint(image, mask, inpaintRadius=3, flags=flag)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Inpaint missing regions in `image` guided by `mask`.

        mask must be:
        - same height & width as image
        - uint8, with 255 indicating missing pixels
        """
        if self.model_loaded and self.model is not None:
            # Real model inference goes here
            # Example pseudocode:
            # inp = preprocess(image, mask)
            # with torch.no_grad():
            #     out = self.model(inp)
            # return postprocess(out)
            raise NotImplementedError("Deep model inference not wired yet.")
        else:
            if not self.use_fallback:
                raise RuntimeError("No model loaded and fallback disabled.")
            return self._opencv_inpaint(image, mask)
