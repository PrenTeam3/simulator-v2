"""Static fixed-position crop finder for A4 working area."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


STATIC_CROP_WIDTH = 2340
STATIC_CROP_HEIGHT = 1630
STATIC_START_X = 1043
STATIC_START_Y = 370


def _compute_crop_bounds(
    image_shape: tuple,
    start_x: int,
    start_y: int,
    crop_width: int,
    crop_height: int,
) -> tuple[int, int, int, int]:
    """Compute clamped crop bounds: (x0, y0, x1, y1)."""
    h, w = image_shape[:2]
    crop_w = min(int(crop_width), w)
    crop_h = min(int(crop_height), h)

    max_x0 = max(0, w - crop_w)
    max_y0 = max(0, h - crop_h)
    x0 = int(np.clip(int(start_x), 0, max_x0))
    y0 = int(np.clip(int(start_y), 0, max_y0))

    x1 = x0 + crop_w
    y1 = y0 + crop_h
    return x0, y0, x1, y1


def crop_static_a4_area(
    image: np.ndarray,
    start_x: int = STATIC_START_X,
    start_y: int = STATIC_START_Y,
    crop_width: int = STATIC_CROP_WIDTH,
    crop_height: int = STATIC_CROP_HEIGHT,
) -> np.ndarray:
    """
    Crop a predefined fixed region from the input image.

    Default is tuned for images around 4608x2592, starting at (1043, 374)
    with size 2340x1630.
    """
    x0, y0, x1, y1 = _compute_crop_bounds(
        image.shape, start_x, start_y, crop_width, crop_height
    )
    return image[y0:y1, x0:x1].copy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Crop static A4 fixed region from image.")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--output-dir",
        default="puzzle_finder/output",
        help="Directory for output image (default: puzzle_finder/output)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=STATIC_CROP_WIDTH,
        help=f"Crop width in pixels (default: {STATIC_CROP_WIDTH})",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=STATIC_CROP_HEIGHT,
        help=f"Crop height in pixels (default: {STATIC_CROP_HEIGHT})",
    )
    parser.add_argument(
        "--x",
        type=int,
        default=STATIC_START_X,
        help=f"Top-left X start in pixels (default: {STATIC_START_X})",
    )
    parser.add_argument(
        "--y",
        type=int,
        default=STATIC_START_Y,
        help=f"Top-left Y start in pixels (default: {STATIC_START_Y})",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Could not load image: {args.image_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x0, y0, x1, y1 = _compute_crop_bounds(
        image.shape, args.x, args.y, args.width, args.height
    )
    cropped = image[y0:y1, x0:x1].copy()
    output_path = output_dir / "a4_static_crop.png"
    cv2.imwrite(str(output_path), cropped)

    print("Static crop created.")
    print(f"Input size: {image.shape[1]}x{image.shape[0]}")
    print(f"Start (x,y): {x0},{y0}")
    print(f"Crop size: {cropped.shape[1]}x{cropped.shape[0]}")
    print(f"Output: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
