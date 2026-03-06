"""Detect and rectify A4 paper regions in images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


A4_ASPECT_RATIO = 297.0 / 210.0  # long side / short side


@dataclass
class A4Detection:
    """Represents a detected A4 sheet in image coordinates."""

    corners: np.ndarray  # shape (4, 2), ordered TL, TR, BR, BL
    contour_area: float
    aspect_ratio: float


def _order_corners(points: np.ndarray) -> np.ndarray:
    """Order corner points clockwise: top-left, top-right, bottom-right, bottom-left."""
    pts = points.astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def _build_white_mask(
    image: np.ndarray,
    debug_dir: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Build a mask that highlights white-ish paper.

    The mask is tuned for white A4 paper on wood backgrounds and tolerates
    dark puzzle pieces lying on the paper.
    """
    debug_path = Path(debug_dir) if debug_dir else None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # White paper is usually bright and low saturation.
    white_mask = cv2.inRange(hsv, (0, 0, 120), (180, 95, 255))
    if debug_path:
        cv2.imwrite(str(debug_path / "02_hsv_white_mask.png"), white_mask)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright_mask = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if debug_path:
        cv2.imwrite(str(debug_path / "03_otsu_bright_mask.png"), bright_mask)

    combined = cv2.bitwise_and(white_mask, bright_mask)
    if debug_path:
        cv2.imwrite(str(debug_path / "04_combined_before_morph.png"), combined)

    # Close holes caused by dark puzzle pieces on top of paper.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, open_kernel, iterations=1)
    if debug_path:
        cv2.imwrite(str(debug_path / "05_mask_final.png"), combined)
    return combined


def detect_a4_area(
    image: np.ndarray,
    min_image_coverage: float = 0.10,
    debug_dir: Optional[Union[str, Path]] = None,
) -> Optional[A4Detection]:
    """
    Detect the dominant A4-like white area.

    Args:
        image: BGR image.
        min_image_coverage: Minimum contour area as fraction of full image.
        debug_dir: If set, save step-by-step debug images (01_input through 09_a4_detected).

    Returns:
        A4Detection if found, else None.
    """
    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path / "01_input.png"), image)

    mask = _build_white_mask(image, debug_dir=debug_dir)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug_path:
        # 06: all contours on mask
        vis_all = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_all, contours, -1, (0, 255, 0), 2)
        cv2.putText(
            vis_all, f"contours: {len(contours)}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
        )
        cv2.imwrite(str(debug_path / "06_contours_all.png"), vis_all)

    if not contours:
        return None

    img_area = float(image.shape[0] * image.shape[1])
    min_area = img_area * float(min_image_coverage)

    contours_min_area = []
    contours_a4_ratio = []

    best_detection = None
    best_score = float("inf")
    best_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        contours_min_area.append(contour)

        rect = cv2.minAreaRect(contour)
        (w, h) = rect[1]
        if w <= 1 or h <= 1:
            continue

        ratio = max(w, h) / min(w, h)
        if ratio < 1.18 or ratio > 1.75:
            continue
        contours_a4_ratio.append(contour)

        # Prefer large contours with A4-like ratio.
        ratio_error = abs(ratio - A4_ASPECT_RATIO)
        area_bonus = area / img_area
        score = ratio_error - 0.35 * area_bonus

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
            best_score = score
            best_contour = contour

    if debug_path:
        # 07: contours passing min_area
        vis_min = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_min, contours_min_area, -1, (0, 255, 0), 2)
        cv2.putText(
            vis_min, f"min_area>={min_area:.0f} ({len(contours_min_area)})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imwrite(str(debug_path / "07_contours_min_area.png"), vis_min)

        # 08: contours passing A4 aspect ratio
        vis_ratio = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_ratio, contours_a4_ratio, -1, (0, 255, 0), 2)
        cv2.putText(
            vis_ratio, f"A4 ratio 1.18-1.75 ({len(contours_a4_ratio)})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        cv2.imwrite(str(debug_path / "08_contours_a4_ratio.png"), vis_ratio)

        # 09: chosen A4 on original image
        vis_final = image.copy()
        if best_contour is not None and best_detection is not None:
            cv2.drawContours(vis_final, [best_contour], -1, (0, 255, 0), 3)
            for idx, pt in enumerate(best_detection.corners.astype(int)):
                cv2.circle(vis_final, tuple(pt), 8, (0, 0, 255), -1)
                cv2.putText(
                    vis_final, str(idx), (int(pt[0]) + 10, int(pt[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )
            cv2.putText(
                vis_final, f"ratio={best_detection.aspect_ratio:.3f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
            )
        else:
            cv2.putText(
                vis_final, "No A4 candidate", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )
        cv2.imwrite(str(debug_path / "09_a4_detected.png"), vis_final)

    return best_detection


def warp_a4_region(image: np.ndarray, detection: A4Detection) -> np.ndarray:
    """
    Perspective-rectify the detected A4 sheet.

    Keeps the detected scale (in pixels) while forcing A4 proportions.
    """
    corners = detection.corners.astype(np.float32)
    tl, tr, br, bl = corners

    top_w = np.linalg.norm(tr - tl)
    bottom_w = np.linalg.norm(br - bl)
    left_h = np.linalg.norm(bl - tl)
    right_h = np.linalg.norm(br - tr)

    observed_w = max(top_w, bottom_w)
    observed_h = max(left_h, right_h)

    if observed_w >= observed_h:
        out_w = int(max(50, round(observed_w)))
        out_h = int(max(50, round(out_w / A4_ASPECT_RATIO)))
    else:
        out_h = int(max(50, round(observed_h)))
        out_w = int(max(50, round(out_h / A4_ASPECT_RATIO)))

    dst = np.array(
        [
            [0, 0],
            [out_w - 1, 0],
            [out_w - 1, out_h - 1],
            [0, out_h - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h))
