"""Detect puzzle pieces in a rectified A4 image."""

from __future__ import annotations

import cv2
import numpy as np


def _detect_background_brightness(gray: np.ndarray) -> float:
    """Sample border pixels to determine if background is light or dark."""
    h, w = gray.shape
    border = min(30, min(h, w) // 10)
    border_pixels = np.concatenate([
        gray[:border, :].flatten(),
        gray[-border:, :].flatten(),
        gray[:, :border].flatten(),
        gray[:, -border:].flatten(),
    ])
    return float(np.median(border_pixels))


A4_WIDTH_MM = 297.0
A4_HEIGHT_MM = 210.0


def _min_area_px(image_shape: tuple, min_area_cm2: float) -> float:
    """Convert a physical area in cm² to pixels² based on A4 image dimensions."""
    h, w = image_shape[:2]
    px_per_cm_x = w / (A4_WIDTH_MM / 10)
    px_per_cm_y = h / (A4_HEIGHT_MM / 10)
    return min_area_cm2 * px_per_cm_x * px_per_cm_y


def detect_pieces(image: np.ndarray, min_area_cm2: float = 5.0):
    """
    Detect puzzle pieces in a rectified A4 image.

    Args:
        image: BGR image (should be the warped A4 region).
        min_area_cm2: Minimum piece area in cm² (default: 5.0 cm²).

    Returns:
        contours: List of contours, one per detected piece.
        thresh: The binary threshold image used for detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    brightness = _detect_background_brightness(gray)

    if brightness > 127:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print(f"Light background detected (brightness: {brightness:.1f})")
    else:
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print(f"Dark background detected (brightness: {brightness:.1f})")

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    min_area = _min_area_px(image.shape, min_area_cm2)
    print(f"Minimum piece area: {min_area_cm2} cm² = {min_area:.0f} px²")

    all_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in all_contours if cv2.contourArea(c) >= min_area]

    print(f"Found {len(all_contours)} total contours, {len(contours)} valid pieces")

    return contours, thresh


def draw_pieces(image: np.ndarray, contours: list) -> np.ndarray:
    """
    Draw detected pieces on a copy of the image.

    Draws a blue outline around each piece, a green dot at its centroid,
    and a piece index label.

    Args:
        image: BGR image to draw on.
        contours: List of contours from detect_pieces().

    Returns:
        Annotated copy of the image.
    """
    out = image.copy()

    for idx, contour in enumerate(contours):
        cv2.drawContours(out, [contour], 0, (255, 0, 0), 2)

        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(out, (cx, cy), 8, (0, 255, 0), -1)
            cv2.putText(out, str(idx), (cx + 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return out
