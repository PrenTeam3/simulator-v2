"""Step 1 — Frame definition.

Defines the fixed puzzle frame (190x128 mm, landscape) and
provides the pixel-to-mm conversion derived from the A4 image width.
"""

from __future__ import annotations

from dataclasses import dataclass


# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

PUZZLE_WIDTH_MM  = 190.0   # horizontal (top / bottom sides)
PUZZLE_HEIGHT_MM = 128.0   # vertical   (left / right sides)
A4_WIDTH_MM      = 297.0   # A4 landscape width — basis for px/mm conversion


# ─────────────────────────────────────────────
#  Frame dataclass
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class PuzzleFrame:
    """Immutable frame descriptor produced by build_frame()."""
    width_mm:   float   # 190.0
    height_mm:  float   # 128.0
    px_per_mm:  float   # a4_image_width_px / 297.0

    def px_to_mm(self, length_px: float) -> float:
        """Convert a pixel length to mm."""
        return length_px / self.px_per_mm

    def mm_to_px(self, length_mm: float) -> float:
        """Convert a mm length to pixels."""
        return length_mm * self.px_per_mm


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

def build_frame(a4_image_width_px: int) -> PuzzleFrame:
    """
    Create a PuzzleFrame from the rectified A4 image width in pixels.

    Args:
        a4_image_width_px: width of the warped A4 image in pixels
                           (comes directly from a4_image.shape[1]).

    Returns:
        PuzzleFrame with width=190mm, height=128mm and computed px_per_mm.
    """
    px_per_mm = a4_image_width_px / A4_WIDTH_MM
    return PuzzleFrame(
        width_mm=PUZZLE_WIDTH_MM,
        height_mm=PUZZLE_HEIGHT_MM,
        px_per_mm=px_per_mm,
    )
