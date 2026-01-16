# app.py
"""
Gradio demo application for the Image Restoration System.

Features:
- Upload image or video
- Optional mask upload for inpainting (image mode)
- Automatic task routing via TaskRouter (heuristic)
- Or manual task selection (override)
- Side-by-side before / after for images
- Downloadable interpolated video output for videos

Notes:
- All model modules include safe fallbacks (OpenCV) so this demo will run
  immediately without pretrained weights.
- The services/pipeline modules manage device selection (CPU/CUDA).
"""

import os
import tempfile
from typing import List, Optional

import numpy as np
import cv2
import gradio as gr

from pipeline import TaskRouter, RestorationPipeline
from utils import (
    load_image,
    save_image,
    load_video_frames,
    save_video,
    compute_psnr,
    compute_ssim,
    get_logger,
)

LOGGER = get_logger("app")

# Instantiate router and pipeline once (reuse)
ROUTER = TaskRouter()
PIPELINE = RestorationPipeline()  # accepts optional device/weights dict if needed


# -------------------------
# Utility conversions
# -------------------------
def _to_bgr_uint8(img) -> np.ndarray:
    """
    Normalize input from Gradio (PIL or numpy RGB) to BGR uint8 for OpenCV/services.
    """
    if img is None:
        raise ValueError("Input image is None")

    # If path supplied (string), use load_image
    if isinstance(img, str):
        return load_image(img)

    arr = np.array(img)  # PIL.Image or numpy
    if arr.ndim == 2:  # grayscale
        # convert to BGR
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        return bgr
    if arr.shape[2] == 4:
        # drop alpha
        arr = arr[..., :3]
    # Gradio/PIL gives RGB; convert to BGR
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr.astype("uint8")


def _to_rgb_uint8_from_bgr(bgr_img: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 (OpenCV) to RGB uint8 for Gradio display.
    """
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return rgb


def _prepare_mask(mask_input) -> Optional[np.ndarray]:
    """
    Convert mask input (PIL/numpy or path) to uint8 binary mask where 255 indicates missing region.
    Returns None if no mask provided.
    """
    if mask_input is None:
        return None

    if isinstance(mask_input, str):
        mask = cv2.imread(mask_input, cv2.IMREAD_GRAYSCALE)
    else:
        arr = np.array(mask_input)
        if arr.ndim == 3:
            mask = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            mask = arr
    if mask is None:
        return None

    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return bin_mask.astype("uint8")


# -------------------------
# Image handler
# -------------------------
def handle_image(
    image,  # PIL or numpy (RGB) from Gradio
    mask,   # optional mask (PIL/numpy)
    auto_route: bool = True,
    manual_tasks: List[str] = None,
):
    """
    Process an uploaded image and return (before_rgb, after_rgb, metrics_dict).
    """
    try:
        LOGGER.info("Received image for processing.")
        img_bgr = _to_bgr_uint8(image)
        mask_bgr = _prepare_mask(mask)
        has_mask = mask_bgr is not None

        if auto_route:
            tasks = ROUTER.route_image(img_bgr, has_mask=has_mask)
            LOGGER.info(f"Auto-routing selected tasks: {tasks}")
        else:
            tasks = manual_tasks or []
            LOGGER.info(f"Manual tasks: {tasks}")

        # run pipeline
        restored_bgr = PIPELINE.process_image(img_bgr, tasks=tasks, mask=mask_bgr)

        # prepare outputs for Gradio (RGB)
        before_rgb = _to_rgb_uint8_from_bgr(img_bgr)
        after_rgb = _to_rgb_uint8_from_bgr(restored_bgr)

        # compute metrics if applicable (we don't have ground-truth generally;
        # here we compute PSNR/SSIM between input and restored for reference)
        try:
            psnr_val = compute_psnr(before_rgb, after_rgb)
            ssim_val = compute_ssim(before_rgb, after_rgb)
        except Exception:
            psnr_val, ssim_val = None, None

        metrics = {
            "psnr (input->restored)": f"{psnr_val:.3f}" if psnr_val not in (None, float("inf")) else str(psnr_val),
            "ssim (input->restored)": f"{ssim_val:.4f}" if ssim_val is not None else "n/a",
            "applied_tasks": ", ".join(tasks) if tasks else "none",
        }

        return before_rgb, after_rgb, metrics

    except Exception as exc:
        LOGGER.exception("Failed to process image")
        return None, None, {"error": str(exc)}


# -------------------------
# Video handler
# -------------------------
def handle_video(
    video_path: str,
    auto_route: bool = True,
    manual_tasks: List[str] = None,
    num_mid_frames: int = 1,
):
    """
    Process uploaded video file path. Returns path to output video and a summary dict.
    """
    try:
        LOGGER.info(f"Received video for processing: {video_path}")
        frames, fps = load_video_frames(video_path)
        if len(frames) == 0:
            raise ValueError("No frames found in the uploaded video.")

        if auto_route:
            tasks = ROUTER.route_video(frames)
            LOGGER.info(f"Auto-routing selected tasks for video: {tasks}")
        else:
            tasks = manual_tasks or []
            LOGGER.info(f"Manual tasks for video: {tasks}")

        # ensure interpolation uses requested mid frames
        out_frames = PIPELINE.process_video(frames, tasks=tasks, num_mid_frames=num_mid_frames)

        # Save to temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.close()

        # Compute new fps: if interpolation added frames, the pipeline already adjusted fps in interpolation model's demo;
        # As a simple approach, set new_fps = fps * (num_mid_frames + 1) if interpolate was applied, else keep fps.
        if "interpolate" in tasks:
            new_fps = fps * (num_mid_frames + 1)
        else:
            new_fps = fps

        save_video(tmp_path, out_frames, fps=new_fps)
        LOGGER.info(f"Saved interpolated/restored video to {tmp_path}")

        summary = {
            "orig_frames": len(frames),
            "out_frames": len(out_frames),
            "orig_fps": float(fps),
            "out_fps": float(new_fps),
            "applied_tasks": ", ".join(tasks) if tasks else "none",
        }

        return tmp_path, summary

    except Exception as exc:
        LOGGER.exception("Failed to process video")
        return None, {"error": str(exc)}


# -------------------------
# Gradio UI
# -------------------------
def build_interface():
    with gr.Blocks(title="Image Restoration System") as demo:
        gr.Markdown("# Image & Video Restoration — Demo")
        gr.Markdown(
            "Upload an image or video. The system will auto-route restoration tasks (denoise, inpaint, colorize, interpolate). "
            "You can also override and pick tasks manually."
        )

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Input Image (RGB)", type="pil")
                    input_mask = gr.Image(label="Optional Mask (white=hole)", type="pil")
                    auto_route_img = gr.Checkbox(value=True, label="Auto route tasks (recommended)")
                    manual_tasks_img = gr.CheckboxGroup(
                        choices=["denoise", "inpaint", "colorize"],
                        label="Manual tasks (used if Auto route is unchecked)",
                        value=[]
                    )
                    run_img = gr.Button("Run Image Restoration")
                    download_after = gr.Button("Download Restored Image")
                with gr.Column(scale=1):
                    before_img = gr.Image(label="Before", type="numpy")
                    after_img = gr.Image(label="After", type="numpy")
                    metrics_output = gr.JSON(label="Metrics / Info")

            def _run_image(img, mask, auto_route, manual_tasks):
                b, a, m = handle_image(img, mask, auto_route=auto_route, manual_tasks=manual_tasks)
                return b, a, m

            run_img.click(
                _run_image,
                inputs=[input_image, input_mask, auto_route_img, manual_tasks_img],
                outputs=[before_img, after_img, metrics_output],
            )

            # download button: generate temp file from 'after_img' content
            def _download_after(arr):
                if arr is None:
                    return None
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                tmp_path = tmp.name
                tmp.close()
                # arr is RGB numpy
                save_image(tmp_path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                return tmp_path

            download_after.click(_download_after, inputs=[after_img], outputs=[gr.File(label="Download Restored Image")])

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_video = gr.Video(label="Input Video (mp4, webm)")
                    auto_route_vid = gr.Checkbox(value=True, label="Auto route tasks (recommended)")
                    manual_tasks_vid = gr.CheckboxGroup(
                        choices=["interpolate", "denoise"],
                        label="Manual tasks (used if Auto route is unchecked)",
                        value=[]
                    )
                    num_mid = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Intermediate frames per pair (num_mid)")
                    run_vid = gr.Button("Run Video Restoration")
                    download_video = gr.Button("Download Restored Video")
                with gr.Column(scale=1):
                    out_video = gr.Video(label="Restored Video")
                    video_info = gr.JSON(label="Video Info / Summary")

            def _run_video(vid_path, auto_route, manual_tasks, num_mid_frames):
                if vid_path is None:
                    return None, {"error": "No video provided"}
                out_path, summary = handle_video(vid_path, auto_route=auto_route, manual_tasks=manual_tasks, num_mid_frames=int(num_mid_frames))
                return out_path, summary

            run_vid.click(
                _run_video,
                inputs=[input_video, auto_route_vid, manual_tasks_vid, num_mid],
                outputs=[out_video, video_info]
            )

            # download simply exposes the same file
            def _download_video_file(path):
                return path

            download_video.click(_download_video_file, inputs=[out_video], outputs=[gr.File(label="Download Restored Video")])

        gr.Markdown("## Notes\n- This demo uses model fallbacks (OpenCV) by default. Replace model weights in `models/*/weights/` to enable deep-model inference.\n- For large models, run locally with a GPU or use Colab with GPU runtime.\n- Logs are printed to console.")

    return demo


if __name__ == "__main__":
    app = build_interface()
    # For local development. Remove share=True in production if not needed.
    app.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=True,
    css=".gradio-container {max-width: 1100px}"
    )

