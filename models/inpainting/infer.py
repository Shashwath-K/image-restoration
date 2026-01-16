"""
CLI tool for image inpainting.

Usage:
python models/inpainting/infer.py \
    --image data/samples/damaged.png \
    --mask  data/samples/mask.png \
    --out   out/inpainted.png
"""

import argparse
from .model import InpaintingModel
from .utils import read_image, read_mask, save_image

def main():
    parser = argparse.ArgumentParser(description="Image inpainting utility")
    parser.add_argument("--image", required=True, help="Path to damaged image")
    parser.add_argument("--mask", required=True, help="Path to mask image")
    parser.add_argument("--out", required=True, help="Path to save output")
    parser.add_argument("--weights", default=None, help="Path to model weights")
    parser.add_argument("--device", default=None, help="cuda or cpu")
    args = parser.parse_args()

    image = read_image(args.image)
    mask = read_mask(args.mask)

    model = InpaintingModel(
        device=args.device,
        weights_path=args.weights,
        use_fallback=True
    )

    result = model.inpaint(image, mask)
    save_image(args.out, result)

    print(f"Inpainted image saved to {args.out}")

if __name__ == "__main__":
    main()
