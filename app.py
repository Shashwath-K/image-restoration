# app.py
"""
Gradio demo application for the Image Restoration System.

Features:
- Premium "Soft" Theme and Layout
- Automatic task routing via TaskRouter
- Manual task selection override
- Side-by-side comparison
- Deep Learning restoration (DnCNN, UNet, ColorNet)
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

# Instantiate router and pipeline
ROUTER = TaskRouter()
# Pipeline automatically uses default weights paths for DnCNN, UNet, ColorNet
PIPELINE = RestorationPipeline()

# -------------------------
# Utility conversions
# -------------------------
def _to_bgr_uint8(img) -> np.ndarray:
    if img is None:
        raise ValueError("Input image is None")
    if isinstance(img, str):
        return load_image(img)
    arr = np.array(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR).astype("uint8")

def _to_rgb_uint8_from_bgr(bgr_img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

def _prepare_mask(mask_input) -> Optional[np.ndarray]:
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
# Handlers
# -------------------------
def handle_image(image, mask, auto_route, manual_tasks):
    try:
        LOGGER.info("Processing Image...")
        img_bgr = _to_bgr_uint8(image)
        mask_bgr = _prepare_mask(mask)
        has_mask = mask_bgr is not None

        if auto_route:
            tasks = ROUTER.route_image(img_bgr, has_mask=has_mask)
        else:
            tasks = manual_tasks or []
            
        restored_bgr = PIPELINE.process_image(img_bgr, tasks=tasks, mask=mask_bgr)
        
        before_rgb = _to_rgb_uint8_from_bgr(img_bgr)
        after_rgb = _to_rgb_uint8_from_bgr(restored_bgr)

        # Metrics
        try:
            psnr = compute_psnr(before_rgb, after_rgb)
            ssim = compute_ssim(before_rgb, after_rgb)
            psnr_str = f"{psnr:.2f}" if psnr else "N/A"
            ssim_str = f"{ssim:.3f}" if ssim else "N/A"
        except:
            psnr_str, ssim_str = "N/A", "N/A"

        info = f"Tasks: {', '.join(tasks) if tasks else 'None'}\nPSNR: {psnr_str} | SSIM: {ssim_str}"
        return after_rgb, info
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        return None, f"Error: {e}"

def handle_video(video, auto_route, manual_tasks, num_mid):
    try:
        if video is None: 
            return None, "No video uploaded."
        
        LOGGER.info(f"Processing Video: {video}")
        frames, fps = load_video_frames(video)
        
        if auto_route:
            tasks = ROUTER.route_video(frames)
        else:
            tasks = manual_tasks or []

        out_frames = PIPELINE.process_video(frames, tasks=tasks, num_mid_frames=int(num_mid))

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.close()

        # Simple fps adjustment
        new_fps = fps * (num_mid + 1) if "interpolate" in tasks else fps
        save_video(tmp_path, out_frames, fps=new_fps)
        
        summary = f"Tasks: {', '.join(tasks)}\nOriginal: {len(frames)} frames @ {fps:.1f}fps\nOutput: {len(out_frames)} frames @ {new_fps:.1f}fps"
        return tmp_path, summary
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        return None, f"Error: {e}"

# -------------------------
# UI Construction
# -------------------------
def build_interface():
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
    )

    with gr.Blocks(theme=theme, title="Restoration Pro") as demo:
        gr.Markdown(
            """
            # 🖼️ AI Image & Video Restoration
            **Deep Learning Powered**: DnCNN (Denoise), U-Net (Inpaint), ColorNet (Colorize).
            """
        )
        
        with gr.Tabs():
            # --- IMAGE TAB ---
            with gr.TabItem("Image Restoration"):
                with gr.Row():
                    # Left Sidebar
                    with gr.Column(scale=1, min_width=300):
                        input_img = gr.Image(label="Input", type="pil", height=300)
                        input_mask = gr.Image(label="Mask (Inpainting)", type="pil", height=150)
                        
                        with gr.Accordion("Settings", open=True):
                            auto_check = gr.Checkbox(value=True, label="Auto-Detect Tasks")
                            manual_check = gr.CheckboxGroup(
                                ["denoise", "inpaint", "colorize"], 
                                label="Manual Override",
                                visible=False
                            )
                            
                        # Toggle visibility based on auto check
                        def toggle_manual(auto):
                            return gr.update(visible=not auto)
                        
                        auto_check.change(toggle_manual, inputs=auto_check, outputs=manual_check)
                        
                        run_btn = gr.Button("✨ Restore Image", variant="primary", size="lg")

                    # Right Display
                    with gr.Column(scale=2):
                        output_img = gr.Image(label="Restored Result", type="numpy", interactive=False)
                        info_box = gr.Textbox(label="Process Info", interactive=False)
                        dl_btn = gr.Button("Download Result")
                        dl_file = gr.File(label="Download", visible=False)

                # Bindings
                run_btn.click(
                    handle_image, 
                    inputs=[input_img, input_mask, auto_check, manual_check],
                    outputs=[output_img, info_box]
                )
                
                def prepare_download(img):
                    if img is None: return None
                    t = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(t, bgr)
                    return gr.update(value=t, visible=True)
                
                dl_btn.click(prepare_download, inputs=output_img, outputs=dl_file)

            # --- VIDEO TAB ---
            with gr.TabItem("Video Enhancement"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        input_vid = gr.Video(label="Input Video")
                        
                        with gr.Accordion("Settings", open=True):
                            auto_vid = gr.Checkbox(value=True, label="Auto-Detect Tasks")
                            manual_vid = gr.CheckboxGroup(
                                ["interpolate", "denoise"],
                                label="Manual Override",
                                visible=False
                            )
                            sl_mid = gr.Slider(1, 4, value=1, step=1, label="Interpolation frames")

                        auto_vid.change(toggle_manual, inputs=auto_vid, outputs=manual_vid)
                        run_vid_btn = gr.Button("🎞️ Enhance Video", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        output_vid = gr.Video(label="Enhanced Result")
                        vid_info = gr.Textbox(label="Process Info")

                run_vid_btn.click(
                    handle_video,
                    inputs=[input_vid, auto_vid, manual_vid, sl_mid],
                    outputs=[output_vid, vid_info]
                )

    return demo

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="127.0.0.1", server_port=7860, share=True)
