"""
models/inpainting/utils.py
Utility functions for image & mask handling.
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

def read_mask(path: str) -> np.ndarray:
    """
    Reads a mask image.
    White (255) = region to be inpainted.
    """
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask
