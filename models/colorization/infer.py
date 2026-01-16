"""
CLI tool for image colorization.

Usage:
python models/colorization/infer.py \
    --image data/samples/gray.png \
    --out   out/colorized.png
"""

import argparse
from .model import ColorizationModel
from .utils import read_image, save_image

def main():
    parser = argparse.ArgumentParser(description="Image colorization utility")
    parser.add_argument("--image", required=True, help="Path to grayscale image")
    parser.add_argument("--out", required=True, help="Path to save output")
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    image = read_image(args.image)

    model = ColorizationModel(
        device=args.device,
        weights_path=args.weights,
        use_fallback=True,
        saturation_boost=1.2
    )

    result = model.colorize(image)
    save_image(args.out, result)

    print(f"Colorized image saved to {args.out}")

if __name__ == "__main__":
    main()
