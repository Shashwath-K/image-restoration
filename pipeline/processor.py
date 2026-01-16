"""
pipeline/processor.py

Executes restoration pipelines using services.
"""

from typing import Optional, List
import numpy as np

from services import (
    DenoiseService,
    InpaintService,
    ColorizeService,
    InterpolateService
)

class RestorationPipeline:
    def __init__(self,
                 device: Optional[str] = None,
                 weights: Optional[dict] = None):
        """
        weights: optional dict mapping task -> weights_path
        Example:
        {
            "denoise": "models/denoising/weights/dncnn.pth",
            "inpaint": "models/inpainting/weights/lama.pth"
        }
        """
        self.weights = weights or {}

        self.denoise_service = DenoiseService(
            device=device,
            weights_path=self.weights.get("denoise")
        )
        self.inpaint_service = InpaintService(
            device=device,
            weights_path=self.weights.get("inpaint")
        )
        self.colorize_service = ColorizeService(
            device=device,
            weights_path=self.weights.get("colorize")
        )
        self.interpolate_service = InterpolateService(
            device=device,
            weights_path=self.weights.get("interpolate")
        )

    # --------------------------------------------------
    # Image pipeline
    # --------------------------------------------------
    def process_image(self,
                      image: np.ndarray,
                      tasks: List[str],
                      mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply a sequence of tasks to an image.
        """
        out = image

        for task in tasks:
            if task == "denoise":
                out = self.denoise_service.restore(out)

            elif task == "inpaint":
                if mask is None:
                    raise ValueError("Mask is required for inpainting.")
                out = self.inpaint_service.restore(out, mask)

            elif task == "colorize":
                out = self.colorize_service.restore(out)

            else:
                raise ValueError(f"Unknown task: {task}")

        return out

    # --------------------------------------------------
    # Video pipeline
    # --------------------------------------------------
    def process_video(self,
                      frames: List[np.ndarray],
                      tasks: List[str],
                      num_mid_frames: int = 1) -> List[np.ndarray]:
        """
        Apply a sequence of tasks to a video (frame list).
        """
        out_frames = frames

        for task in tasks:
            if task == "interpolate":
                out_frames = self.interpolate_service.restore(
                    out_frames,
                    num_mid_frames=num_mid_frames
                )

            elif task == "denoise":
                out_frames = [
                    self.denoise_service.restore(f) for f in out_frames
                ]

            else:
                raise ValueError(f"Unknown task for video: {task}")

        return out_frames
