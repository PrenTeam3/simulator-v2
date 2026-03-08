"""CLI for standalone A4 detection testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

if __package__ in (None, ""):
    from a4_finder import detect_a4_area, warp_a4_region
else:
    from .a4_finder import detect_a4_area, warp_a4_region


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect and rectify A4 area from an image.")
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--output-dir",
        default="puzzle_finder/output",
        help="Directory for output and debug images (default: puzzle_finder/output)",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Do not save step-by-step debug images",
    )
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Could not load image: {args.image_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = None if args.no_debug else output_dir

    detection = detect_a4_area(image, debug_dir=debug_dir)
    if detection is None:
        print("No A4-like area detected.")
        if debug_dir:
            print(f"Check debug images in: {output_dir}")
        return 2

    # Save annotated original
    annotated = image.copy()
    corners = detection.corners.astype(int)
    cv2.polylines(annotated, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
    for idx, pt in enumerate(corners):
        cv2.circle(annotated, tuple(pt), 6, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            str(idx),
            (int(pt[0]) + 8, int(pt[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

    rectified = warp_a4_region(image, detection)

    annotated_path = output_dir / "a4_detected.png"
    rectified_path = output_dir / "a4_rectified.png"
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(rectified_path), rectified)

    print("A4 area detected.")
    print(f"Aspect ratio: {detection.aspect_ratio:.3f}")
    print(f"Contour area: {detection.contour_area:.0f}px")
    print(f"Output: {output_dir}")
    print(f"  Annotated: {annotated_path.name}")
    print(f"  Rectified: {rectified_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
