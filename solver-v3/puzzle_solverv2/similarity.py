"""Step 4 — Pairwise piece similarity analysis.

Compares all piece pairs to detect potential duplicates.
No hard yes/no — only a similarity score is recorded.
The score is used later in Step 6 (tree search) to skip symmetric combinations.

Two-stage comparison (cheapest first):
  1. Area difference — if > 5%: not a duplicate, skip
  2. Hu Moments (cv2.matchShapes) — rotation-invariant shape comparison
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────
#  Thresholds (from algorithm doc)
# ─────────────────────────────────────────────

AREA_TOLERANCE      = 0.05   # ±5%  — if area diff exceeds this, skip shape check
SHAPE_THRESHOLD     = 0.10   # 0.10 — Hu Moments score below this → likely duplicate


# ─────────────────────────────────────────────
#  Data structure
# ─────────────────────────────────────────────

@dataclass
class SimilarityResult:
    piece_a:         int
    piece_b:         int
    area_diff_pct:   float           # relative area difference (0.0–1.0)
    shape_score:     Optional[float] # Hu Moments score (None = area check failed)
    likely_duplicate: bool

    def __repr__(self) -> str:
        if self.shape_score is None:
            return (f"Pair(P{self.piece_a}, P{self.piece_b}): "
                    f"area_diff={self.area_diff_pct*100:.1f}% → skipped (not similar)")
        return (f"Pair(P{self.piece_a}, P{self.piece_b}): "
                f"area_diff={self.area_diff_pct*100:.1f}%  "
                f"shape_score={self.shape_score:.4f}  "
                f"likely_duplicate={self.likely_duplicate}")


# ─────────────────────────────────────────────
#  Analysis
# ─────────────────────────────────────────────

def analyze_similarity(contours: list) -> list[SimilarityResult]:
    """
    Compare all piece pairs for similarity.

    Args:
        contours: raw contours from detect_pieces() — one per piece,
                  in OpenCV format (used directly for contourArea and matchShapes)

    Returns:
        List of SimilarityResult, one per unique pair.
    """
    areas = [cv2.contourArea(c) for c in contours]
    results = []

    for idx_a, idx_b in combinations(range(len(contours)), 2):
        area_a = areas[idx_a]
        area_b = areas[idx_b]

        # Stage 1: area check (cheap)
        area_diff = abs(area_a - area_b) / max(area_a, area_b)
        if area_diff > AREA_TOLERANCE:
            results.append(SimilarityResult(
                piece_a=idx_a,
                piece_b=idx_b,
                area_diff_pct=area_diff,
                shape_score=None,
                likely_duplicate=False,
            ))
            continue

        # Stage 2: Hu Moments shape comparison (rotation-invariant)
        shape_score = cv2.matchShapes(
            contours[idx_a], contours[idx_b],
            cv2.CONTOURS_MATCH_I1, 0,
        )

        results.append(SimilarityResult(
            piece_a=idx_a,
            piece_b=idx_b,
            area_diff_pct=area_diff,
            shape_score=shape_score,
            likely_duplicate=shape_score < SHAPE_THRESHOLD,
        ))

    return results


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def log_similarity(results: list[SimilarityResult]) -> None:
    """Print a structured summary of the similarity analysis."""
    duplicates = [r for r in results if r.likely_duplicate]

    for r in results:
        if r.shape_score is None:
            print(f"  P{r.piece_a} vs P{r.piece_b}: "
                  f"area_diff={r.area_diff_pct*100:.1f}%  → too different, skipped")
        else:
            flag = " *** LIKELY DUPLICATE ***" if r.likely_duplicate else ""
            print(f"  P{r.piece_a} vs P{r.piece_b}: "
                  f"area_diff={r.area_diff_pct*100:.1f}%  "
                  f"shape_score={r.shape_score:.4f}{flag}")

    if duplicates:
        print(f"\n  Likely duplicates found: {len(duplicates)}")
        for r in duplicates:
            print(f"    P{r.piece_a} ≈ P{r.piece_b}  (score={r.shape_score:.4f})")
    else:
        print(f"\n  No likely duplicates found")
