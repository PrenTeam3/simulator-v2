"""Utilities to convert pixel points to A4 millimeter coordinates."""

from __future__ import annotations

import argparse
from typing import Tuple


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def convert_to_A4_cords(
    image_size: Tuple[int, int],
    point_px: Tuple[float, float],
    a4_size_mm: Tuple[float, float] = (A4_WIDTH_MM, A4_HEIGHT_MM),
    clamp: bool = False,
) -> Tuple[float, float]:
    """
    Convert a point from pixel coordinates to A4 millimeter coordinates.

    Args:
        image_size: (width_px, height_px) of the image.
        point_px: (x_px, y_px) point in the image.
        a4_size_mm: (width_mm, height_mm), defaults to A4 (210, 297).
        clamp: If True, clamp point to image bounds before conversion.

    Returns:
        (x_mm, y_mm) point in real-world A4 millimeters.
    """
    width_px, height_px = image_size
    x_px, y_px = point_px
    width_mm, height_mm = a4_size_mm

    if width_px <= 0 or height_px <= 0:
        raise ValueError("image_size must contain positive width and height")
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("a4_size_mm must contain positive width and height")

    if clamp:
        x_px = min(max(x_px, 0.0), float(width_px))
        y_px = min(max(y_px, 0.0), float(height_px))

    x_mm = (float(x_px) / float(width_px)) * float(width_mm)
    y_mm = (float(y_px) / float(height_px)) * float(height_mm)

    return x_mm, y_mm


def main() -> int:
    """Small CLI to convert one pixel point to A4 millimeters."""
    parser = argparse.ArgumentParser(description="Convert pixel point to A4 mm coordinates.")
    parser.add_argument("--image-width", type=int, required=True, help="Image width in pixels")
    parser.add_argument("--image-height", type=int, required=True, help="Image height in pixels")
    parser.add_argument("--x", type=float, required=True, help="Point X in pixels")
    parser.add_argument("--y", type=float, required=True, help="Point Y in pixels")
    parser.add_argument("--clamp", action="store_true", help="Clamp point to image bounds")
    args = parser.parse_args()

    x_mm, y_mm = convert_to_A4_cords(
        image_size=(args.image_width, args.image_height),
        point_px=(args.x, args.y),
        clamp=args.clamp,
    )

    print(f"Input point (px): ({args.x}, {args.y})")
    print(f"A4 point (mm): ({x_mm:.2f}, {y_mm:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
