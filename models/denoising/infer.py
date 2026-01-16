"""
CLI tool for image denoising.

Usage:
python models/denoising/infer.py \
    --image data/samples/noisy.png \
    --out   out/denoised.png
"""

import argparse
from .model import DenoisingModel
from .utils import read_image, save_image

def main():
    parser = argparse.ArgumentParser(description="Image denoising utility")
    parser.add_argument("--image", required=True, help="Path to noisy image")
    parser.add_argument("--out", required=True, help="Path to save output")
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    image = read_image(args.image)

    model = DenoisingModel(
        device=args.device,
        weights_path=args.weights,
        use_fallback=True
    )

    result = model.denoise(image)
    save_image(args.out, result)

    print(f"Denoised image saved to {args.out}")

if __name__ == "__main__":
    main()
