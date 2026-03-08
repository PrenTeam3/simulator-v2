"""Detect and rectify A4 area using a green border contour."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

if __package__ in (None, ""):
    from a4_finder import A4_ASPECT_RATIO, A4Detection, _order_corners, warp_a4_region
else:
    from .a4_finder import A4_ASPECT_RATIO, A4Detection, _order_corners, warp_a4_region


def _build_green_mask(
    image: np.ndarray,
    debug_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Build a binary mask for green border markings.

    The range is intentionally broad to tolerate lighting differences.
    """
    debug_path = Path(debug_dir) if debug_dir else None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 45, 40], dtype=np.uint8)
    upper_green = np.array([95, 255, 255], dtype=np.uint8)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Connect fragmented border segments while suppressing tiny noise.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    if debug_path:
        cv2.imwrite(str(debug_path / "02_green_mask.png"), green_mask)
    return green_mask


def detect_a4_border_area(
    image: np.ndarray,
    min_image_coverage: float = 0.03,
    debug_dir: Optional[Union[str, Path]] = None,
) -> Optional[A4Detection]:
    """
    Detect the dominant A4 region framed by a green border.

    Args:
        image: BGR image.
        min_image_coverage: Minimum contour area as fraction of full image.
        debug_dir: If set, save debug images.

    Returns:
        A4Detection if found, else None.
    """
    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path / "01_input.png"), image)

    mask = _build_green_mask(image, debug_dir=debug_dir)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or hierarchy is None:
        return None
    hierarchy = hierarchy[0]

    img_area = float(image.shape[0] * image.shape[1])
    min_area = img_area * float(min_image_coverage)

    best_detection = None
    best_contour = None
    best_score = float("inf")
    contours_min_area = []
    contours_a4_ratio = []
    contours_inner = []
    contours_outer = []

    def _consider_contour(contour: np.ndarray, contour_weight: float) -> None:
        nonlocal best_detection, best_contour, best_score

        area = cv2.contourArea(contour)
        if area < min_area:
            return
        contours_min_area.append(contour)

        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        if w <= 1 or h <= 1:
            return

        ratio = max(w, h) / min(w, h)
        if ratio < 1.18 or ratio > 1.75:
            return
        contours_a4_ratio.append(contour)

        ratio_error = abs(ratio - A4_ASPECT_RATIO)
        area_bonus = area / img_area
        score = ratio_error - 0.4 * area_bonus + contour_weight

        if score < best_score:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
            else:
                corners = cv2.boxPoints(rect)

            best_detection = A4Detection(
                corners=_order_corners(corners),
                contour_area=float(area),
                aspect_ratio=float(ratio),
            )
            best_contour = contour
            best_score = score

    inner_indices = []
    outer_indices = []
    for idx, relation in enumerate(hierarchy):
        parent_idx = int(relation[3])
        if parent_idx >= 0:
            inner_indices.append(idx)
        else:
            outer_indices.append(idx)

    # Prefer hole contours (inside border), then fallback to outer border if needed.
    for idx in inner_indices:
        contour = contours[idx]
        contours_inner.append(contour)
        _consider_contour(contour, contour_weight=0.0)

    if best_detection is None:
        for idx in outer_indices:
            contour = contours[idx]
            contours_outer.append(contour)
            _consider_contour(contour, contour_weight=0.06)

    if debug_path:
        vis_all = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_all, contours, -1, (0, 255, 0), 2)
        cv2.putText(
            vis_all,
            f"contours: {len(contours)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(debug_path / "03_contours_all.png"), vis_all)

        vis_inner_outer = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if contours_inner:
            cv2.drawContours(vis_inner_outer, contours_inner, -1, (0, 255, 255), 2)
        if contours_outer:
            cv2.drawContours(vis_inner_outer, contours_outer, -1, (255, 0, 0), 2)
        cv2.putText(
            vis_inner_outer,
            f"inner={len(contours_inner)} outer={len(contours_outer)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(debug_path / "03b_inner_outer.png"), vis_inner_outer)

    if debug_path:
        vis_min = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_min, contours_min_area, -1, (0, 255, 0), 2)
        cv2.putText(
            vis_min,
            f"min_area>={min_area:.0f} ({len(contours_min_area)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(debug_path / "04_contours_min_area.png"), vis_min)

        vis_ratio = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_ratio, contours_a4_ratio, -1, (0, 255, 0), 2)
        cv2.putText(
            vis_ratio,
            f"A4 ratio 1.18-1.75 ({len(contours_a4_ratio)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(debug_path / "05_contours_a4_ratio.png"), vis_ratio)

        vis_final = image.copy()
        if best_contour is not None and best_detection is not None:
            cv2.drawContours(vis_final, [best_contour], -1, (0, 255, 0), 3)
            for idx, pt in enumerate(best_detection.corners.astype(int)):
                cv2.circle(vis_final, tuple(pt), 8, (0, 0, 255), -1)
                cv2.putText(
                    vis_final,
                    str(idx),
                    (int(pt[0]) + 10, int(pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
            cv2.putText(
                vis_final,
                f"ratio={best_detection.aspect_ratio:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                vis_final,
                "No A4 candidate",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(str(debug_path / "06_a4_detected.png"), vis_final)

    return best_detection


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and rectify A4 area from green border."
    )
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--output-dir",
        default="solver-v2/puzzle_finder/output/border",
        help="Directory for output and debug images (default: solver-v2/puzzle_finder/output/border)",
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

    detection = detect_a4_border_area(image, debug_dir=debug_dir)
    if detection is None:
        print("No A4-like green border detected.")
        if debug_dir:
            print(f"Check debug images in: {output_dir}")
        return 2

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

    annotated_path = output_dir / "a4_border_detected.png"
    rectified_path = output_dir / "a4_border_rectified.png"
    cv2.imwrite(str(annotated_path), annotated)
    cv2.imwrite(str(rectified_path), rectified)

    print("A4 green border detected.")
    print(f"Aspect ratio: {detection.aspect_ratio:.3f}")
    print(f"Contour area: {detection.contour_area:.0f}px")
    print(f"Output: {output_dir}")
    print(f"  Annotated: {annotated_path.name}")
    print(f"  Rectified: {rectified_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
