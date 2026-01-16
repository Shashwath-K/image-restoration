"""
models/colorization/model.py

Wrapper for grayscale-to-color image conversion.
Primary target: DeOldify / ChromaGAN.
Fallback: OpenCV pseudo-color mapping.
"""

from typing import Optional
import os
import cv2
import numpy as np
import torch

class ColorizationModel:
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
                print(f"[ColorizationModel] Failed to load weights: {e}")
                self.model_loaded = False
        else:
             print(f"[ColorizationModel] Weights not found at {weights_path}. Model will not be loaded.")

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------
    def _load_weights(self, path: str):
        """
        Load ColorNet weights.
        """
        from .eccv16 import ColorNet
        self.model = ColorNet()
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[ColorizationModel] Loaded weights from {path}")

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------
    def colorize(self, image: np.ndarray) -> np.ndarray:
        """
        Colorize an image.

        image : BGR uint8 numpy array
        returns: BGR uint8 color image
        """
        # 0. Check if image is already color (Pre-check)
        if self._is_already_color(image):
            print("[ColorizationModel] Image detected as color. Skipping colorization.")
            return image

        # 1. Prepare Input (BGR -> Lab)
        # Resize for model if needed? Let's just run on full res or rescale.
        # For simplicity, we'll process at input resolution (assuming standard sizes).
        # Normalization: L channel 0-100 -> 0-1 or -1 to 1?
        # Standard: 0-100.
        
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0] # 0-255 in OpenCV uint8 representation (scale L=100 -> 255)
        
        if self.model_loaded and self.model is not None:
             # Preprocess l_channel
             # OpenCV L is 0-255. Model likely expects normalized float.
             l_norm = l_channel.astype(np.float32) / 255.0 # 0-1
             # Center to -0.5 to 0.5? Or 0-1.
             # AutoEncoder usually likes centered.
             l_norm = (l_norm - 0.5) * 2.0 # -1 to 1
             
             # (1, H, W) -> (B, 1, H, W)
             inp_tensor = torch.from_numpy(l_norm).float().unsqueeze(0).unsqueeze(0)
             inp_tensor = inp_tensor.to(self.device)

             with torch.no_grad():
                 # Model outputs ab in -1 to 1 range
                 ab_tensor = self.model(inp_tensor)

             # Postprocess ab
             ab_out = ab_tensor.squeeze(0).cpu().numpy() # (2, H, W)
             ab_out = ab_out.transpose(1, 2, 0) # (H, W, 2)
             
             # Scale -1..1 -> OpenCV a,b range.
             # OpenCV Lab: L=0..255, a=0..255, b=0..255 (offset 128)
             # Real Lab: a,b approx -128..127.
             # OpenCV maps this to 0-255 by adding 128.
             ab_out = (ab_out * 128.0) + 128.0 
             ab_out = np.clip(ab_out, 0, 255).astype(np.uint8)
             
             # Combine L + predicted ab
             # Use original L channel to preserve crispness
             lab_out = np.concatenate([l_channel[:, :, np.newaxis], ab_out], axis=2)
             
             bgr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
             return bgr_out
        else:
             raise RuntimeError("Colorization model not loaded. Cannot perform colorization.")

    def _is_already_color(self, image: np.ndarray, threshold: float = 5.0) -> bool:
        """
        Check if image is effectively color.
        Computes mean difference between channels.
        """
        if len(image.shape) < 3 or image.shape[2] != 3:
            return False
            
        b, g, r = cv2.split(image)
        # Calculate absolute difference between channels
        diff_bg = np.mean(np.abs(b - g))
        diff_gr = np.mean(np.abs(g - r))
        
        # If differences are significant, it's color.
        # Threshold 5.0 is standard low bar.
        return (diff_bg + diff_gr) > threshold
