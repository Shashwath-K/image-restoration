"""
models/denoising/utils.py
Utility helpers for image IO.
"""

import cv2
import numpy as np
from pathlib import Path

def read_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def save_image(path: str, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
