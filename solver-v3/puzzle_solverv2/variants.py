"""Step 3 — Variant generation per piece.

For each piece, reads the already-classified corner_pairs and straight_edges
from Step 2 and produces all meaningful usage variants:

    corner variant — two edges that together form a 90° frame corner
    edge variant   — a single outside edge (piece sits along one frame side)

Non-adjacent edge combinations are intentionally excluded for now.
(See algorithm doc: deferred to a later step.)
"""

from __future__ import annotations

from dataclasses import dataclass
from puzzle_solverv2.border_info import PieceBorderInfo


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────

@dataclass
class Variant:
    piece_idx: int
    type:      str          # 'corner' or 'edge'
    edges:     list[dict]   # 1 seg dict (edge) or 2 seg dicts (corner)

    @property
    def total_length_mm(self) -> float:
        return sum(e['length_mm'] for e in self.edges)

    def __repr__(self) -> str:
        seg_ids = " + ".join(e['seg_id'] for e in self.edges)
        return (f"Variant(piece={self.piece_idx}, type={self.type}, "
                f"segs=[{seg_ids}], length={self.total_length_mm:.1f}mm)")


@dataclass
class PieceVariants:
    piece_idx: int
    variants:  list[Variant]


# ─────────────────────────────────────────────
#  Generation
# ─────────────────────────────────────────────

def generate_variants(pieces: list[PieceBorderInfo]) -> list[PieceVariants]:
    """
    Generate all usage variants for each piece.

    For each piece:
      - One corner variant per corner_pair (a, b)
      - One edge variant per straight_edge

    Args:
        pieces: output of extract_border_info() from Step 2

    Returns:
        List of PieceVariants, one per piece.
    """
    result = []

    for piece in pieces:
        variants = []

        # Corner variants — read directly from already-detected corner_pairs
        for seg_a, seg_b in piece.corner_pairs:
            variants.append(Variant(
                piece_idx=piece.piece_idx,
                type='corner',
                edges=[seg_a, seg_b],
            ))

        # Edge variants — each outside segment individually
        for seg in piece.straight_edges:
            variants.append(Variant(
                piece_idx=piece.piece_idx,
                type='edge',
                edges=[seg],
            ))

        result.append(PieceVariants(
            piece_idx=piece.piece_idx,
            variants=variants,
        ))

    return result


# ─────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────

def log_variants(pieces_variants: list[PieceVariants]) -> None:
    """Print a structured summary of all generated variants."""
    total = sum(len(pv.variants) for pv in pieces_variants)

    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        edge_variants   = [v for v in pv.variants if v.type == 'edge']

        print(f"  Piece {pv.piece_idx}: {len(pv.variants)} variant(s)")

        for v in corner_variants:
            seg_ids = " + ".join(e['seg_id'] for e in v.edges)
            lengths = " + ".join(f"{e['length_mm']:.1f}mm" for e in v.edges)
            print(f"    [corner] {seg_ids}  ({lengths}  = {v.total_length_mm:.1f}mm)")

        for v in edge_variants:
            seg_id = v.edges[0]['seg_id']
            length = v.edges[0]['length_mm']
            print(f"    [edge]   {seg_id}  ({length:.1f}mm)")

    print(f"\n  Total variants: {total}")
