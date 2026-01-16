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
                print(f"[InpaintingModel] Failed to load weights: {e}")
                self.model_loaded = False
        else:
             print(f"[InpaintingModel] Weights not found at {weights_path}. Model will not be loaded.")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_weights(self, path: str):
        """
        Load UNet model weights.
        """
        from .unet import UNet
        self.model = UNet(n_channels=4, n_classes=3, bilinear=True)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()
        print(f"[InpaintingModel] Loaded weights from {path}")

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
             # Preprocess
             # Normalize image to 0-1
             img_norm = image.astype(np.float32) / 255.0
             # Normalize mask to 0-1 (0=valid, 1=missing/target)
             mask_norm = mask.astype(np.float32) / 255.0
             
             # Expand mask dim: (H, W) -> (H, W, 1)
             if mask_norm.ndim == 2:
                 mask_norm = mask_norm[:, :, np.newaxis]
                 
             # Concatenate along channel dim: (H, W, 3) + (H, W, 1) -> (H, W, 4)
             inp = np.concatenate([img_norm, mask_norm], axis=2)
             
             # HWC -> CHW -> Batch
             inp_tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float().unsqueeze(0)
             inp_tensor = inp_tensor.to(self.device)

             with torch.no_grad():
                 out_tensor = self.model(inp_tensor)

             # Postprocess
             out_tensor = out_tensor.squeeze(0).cpu()
             out_img = out_tensor.numpy().transpose(1, 2, 0)
             out_img = np.clip(out_img * 255.0, 0, 255).astype(np.uint8)
             
             # Composite back: Only replace masked regions? 
             # Usually standard inpainting practice, but if model outputs full image, we can just return it.
             # Let's trust the model output for consistency, or we can composite.
             # Composite is safer if model is weak on valid pixels.
             # result = (mask > 127) * out_img + (mask <= 127) * image
             # But here let's just return the model output as it's a "restoration" model.
             return out_img
        else:
             raise RuntimeError("Inpainting model (UNet) not loaded. Cannot perform inpainting.")
