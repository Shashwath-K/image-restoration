from typing import List, Tuple
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

def read_image(path: str) -> np.ndarray:
    """Read image as BGR uint8 numpy array (OpenCV default)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def save_image(path: str, img: np.ndarray) -> None:
    """Save BGR uint8 image to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)

def read_video_frames(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Read all frames from a video file.
    Returns (frames_list, fps)
    Frames are returned as BGR uint8 numpy arrays.
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

def save_video(path: str, frames: List[np.ndarray], fps: float = 30.0, codec: str = "mp4v") -> None:
    """
    Save frames (BGR uint8) to a video file.
    Default codec 'mp4v' produces .mp4 on most systems.
    """
    if len(frames) == 0:
        raise ValueError("No frames to save.")
    h, w = frames[0].shape[:2]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

def frames_to_tensor(frames: List[np.ndarray]) -> np.ndarray:
    """
    Convert list of HxWxC uint8 BGR frames to numpy float32 array
    shaped (N, C, H, W) in RGB order normalized to [0,1].
    """
    out = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        arr = rgb.astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # C, H, W
        out.append(arr)
    return np.stack(out, axis=0)

def tensor_to_frames(tensor: np.ndarray) -> List[np.ndarray]:
    """
    Convert (N, C, H, W) float32 in [0,1] RGB to list of BGR uint8 frames.
    """
    frames = []
    for i in range(tensor.shape[0]):
        arr = tensor[i]
        arr = np.clip(arr * 255.0, 0, 255).astype("uint8")
        arr = np.transpose(arr, (1, 2, 0))  # H, W, C
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames
