"""
pipeline/router.py

Decision logic for selecting restoration tasks.
"""

import numpy as np
import cv2

class TaskRouter:
    """
    Lightweight heuristic-based router.
    Can be upgraded later to a learned classifier.
    """

    def __init__(self,
                 noise_threshold: float = 15.0,
                 grayscale_threshold: float = 5.0):
        self.noise_threshold = noise_threshold
        self.grayscale_threshold = grayscale_threshold

    # --------------------------------------------------
    # Detection helpers
    # --------------------------------------------------
    @staticmethod
    def _estimate_noise(image: np.ndarray) -> float:
        """
        Estimate noise using Laplacian variance.
        Higher variance => sharper image, lower => blur/noise.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score

    @staticmethod
    def _is_grayscale(image: np.ndarray) -> bool:
        """
        Check if image is effectively grayscale.
        """
        if len(image.shape) < 3:
            return True
        b, g, r = cv2.split(image)
        diff_bg = np.mean(np.abs(b - g))
        diff_gr = np.mean(np.abs(g - r))
        return (diff_bg + diff_gr) < 5.0

    # --------------------------------------------------
    # Public routing API
    # --------------------------------------------------
    def route_image(self,
                    image: np.ndarray,
                    has_mask: bool = False) -> list[str]:
        """
        Decide which tasks to apply to an image.
        Returns ordered list of task names.
        """
        tasks = []

        # 1. Denoising first
        noise_score = self._estimate_noise(image)
        if noise_score < self.noise_threshold:
            tasks.append("denoise")

        # 2. Inpainting
        if has_mask:
            tasks.append("inpaint")

        # 3. Colorization
        if self._is_grayscale(image):
            tasks.append("colorize")

        return tasks

    def route_video(self,
                    frames: list[np.ndarray]) -> list[str]:
        """
        Decide tasks for video sequences.
        """
        tasks = []

        # Always interpolate first for broken sequences
        tasks.append("interpolate")

        # Optional: check first frame for noise
        noise_score = self._estimate_noise(frames[0])
        if noise_score < self.noise_threshold:
            tasks.append("denoise")

        return tasks
