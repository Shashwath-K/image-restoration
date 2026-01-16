"""
Simple CLI for frame interpolation demo.

Examples:
# Interpolate two images and save the single intermediate frame (num_mid=1):
python models/interpolation/infer.py --frame1 path/to/a.png --frame2 path/to/b.png --out_dir out --num_mid 1

# Interpolate a video, inserting 1 intermediate between each frame
python models/interpolation/infer.py --input_video data/samples/sample.mp4 --out_video out/interpolated.mp4 --num_mid 1
"""

import argparse
from pathlib import Path
from .model import InterpolationModel
from .utils import read_image, save_image, read_video_frames, save_video

def interp_images(frame1_path, frame2_path, out_dir, num_mid=1, weights=None, device=None):
    f1 = read_image(frame1_path)
    f2 = read_image(frame2_path)
    model = InterpolationModel(device=device, weights_path=weights, fallback_blend=True)
    frames = [f1, f2]
    out_frames = model.interpolate_sequence(frames, num_mid=num_mid)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, fr in enumerate(out_frames):
        save_image(out_dir / f"frame_{idx:04d}.png", fr)
    print(f"Saved {len(out_frames)} frames to {out_dir}")

def interp_video(input_video, out_video, num_mid=1, weights=None, device=None):
    frames, fps = read_video_frames(input_video)
    model = InterpolationModel(device=device, weights_path=weights, fallback_blend=True)
    out_frames = model.interpolate_sequence(frames, num_mid=num_mid)
    # Adjust fps accordingly: if we add num_mid frames between each pair, new_fps = fps * (num_mid + 1)
    new_fps = fps * (num_mid + 1)
    save_video(out_video, out_frames, fps=new_fps)
    print(f"Saved interpolated video to {out_video} with fps={new_fps:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Interpolation demo (fallback to blending if no weights).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_video", type=str, help="Path to input video.")
    group.add_argument("--frame1", type=str, help="Path to first image frame.")
    parser.add_argument("--frame2", type=str, help="Path to second image frame (required if --frame1 given).")
    parser.add_argument("--out_video", type=str, help="Path to output video (for video mode).")
    parser.add_argument("--out_dir", type=str, default="out/frames", help="Directory for output frames (image mode).")
    parser.add_argument("--num_mid", type=int, default=1, help="Number of intermediate frames between each pair.")
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (optional).")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu).")
    args = parser.parse_args()

    if args.input_video:
        if not args.out_video:
            raise SystemExit("When --input_video is used you must supply --out_video")
        interp_video(args.input_video, args.out_video, num_mid=args.num_mid, weights=args.weights, device=args.device)
    else:
        if not args.frame2:
            raise SystemExit("--frame2 is required when using --frame1")
        interp_images(args.frame1, args.frame2, args.out_dir, num_mid=args.num_mid, weights=args.weights, device=args.device)

if __name__ == "__main__":
    main()