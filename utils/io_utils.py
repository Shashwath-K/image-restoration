"""
utils/io_utils.py

Unified I/O helpers for images and videos.
"""

from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path

# -----------------------------
# Image helpers
# -----------------------------
def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def save_image(path: str, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)

# -----------------------------
# Video helpers
# -----------------------------
def load_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Returns:
      frames: list of BGR uint8 frames
      fps   : frames per second
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames, fps

def save_video(path: str,
               frames: List[np.ndarray],
               fps: float = 30.0,
               codec: str = "mp4v") -> None:
    if not frames:
        raise ValueError("No frames to save.")

    h, w = frames[0].shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))

    for f in frames:
        writer.write(f)

    writer.release()
