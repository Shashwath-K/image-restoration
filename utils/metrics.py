"""
utils/metrics.py

Objective image quality metrics.
"""

import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

# --------------------------------------------------
# PSNR
# --------------------------------------------------
def compute_psnr(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    Images must be uint8 BGR with same shape.
    """
    ref_f = ref.astype("float32")
    pred_f = pred.astype("float32")

    mse = np.mean((ref_f - pred_f) ** 2)
    if mse == 0:
        return float("inf")

    max_i = 255.0
    psnr = 10.0 * np.log10((max_i ** 2) / mse)
    return psnr

# --------------------------------------------------
# SSIM
# --------------------------------------------------
def compute_ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    Operates on grayscale for stability.
    """
    ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(ref_gray, pred_gray, full=True)
    return float(score)
