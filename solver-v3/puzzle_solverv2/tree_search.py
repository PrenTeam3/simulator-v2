"""Placement debug — visual check of TL and BL starting positions."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_solverv2.frame import PuzzleFrame, PUZZLE_HEIGHT_MM
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants
from puzzle_solverv2._placement_types import (
    Candidate,
    _SECOND_CONFIGS,
    _SIDE_TOLERANCE,
    empty_occupancy,
    occ_add,
    occ_add_candidate,
)
from puzzle_solverv2._placement_geometry import (
    _place_contour,
    _build_candidates,
)
from puzzle_solverv2._placement_canvas import (
    COLORS,
    _make_canvas,
    _draw_piece,
    _draw_segments,
    _mark_corner,
)
from puzzle_solverv2._placement_drawing import (
    _draw_verdict,
    _draw_progress_bar,
    _draw_legend,
    _side_lines,
    _candidate_lines,
)


# ─────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────

def _draw_candidate(
    canvas:         "np.ndarray",
    ox:             int,
    oy:             int,
    cand:           Candidate,
    end_pos:        str,
    frame:          PuzzleFrame,
    side_name:      str,
    bar_offset:     float,
    bar_color_base: tuple,
    target:         float,
) -> None:
    """Draw a candidate piece + segment markers + progress bar onto canvas."""
    color = COLORS[cand.pv.piece_idx % len(COLORS)]
    _draw_piece(canvas, cand.placed_mm, ox, oy, color, cand.label)
    if cand.seg_h is not None and cand.seg_v is not None:
        _draw_segments(canvas, cand.seg_h, cand.seg_v, end_pos, frame.px_per_mm, ox, oy)
        _mark_corner(canvas, end_pos, ox, oy)
    _draw_progress_bar(canvas, ox, oy, side_name,
                       bar_offset, cand.seg_len,
                       bar_color_base, color, target)


# ─────────────────────────────────────────────
#  Public visualisation functions
# ─────────────────────────────────────────────

def visualize_start_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
) -> None:
    """For each piece with a corner variant, save start.png at TL and BL."""
    border_by_idx = {p.piece_idx: p for p in pieces_border}

    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        pb          = border_by_idx[pv.piece_idx]
        contour_raw = corners_list[pv.piece_idx]['contour_flat']
        color       = COLORS[pv.piece_idx % len(COLORS)]

        for v in corner_variants:
            for position in ('TL', 'BL'):
                seg_h, seg_v = (v.edges[0], v.edges[1]) if position == 'TL' else (v.edges[1], v.edges[0])
                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]

                occ_1 = empty_occupancy()
                occ_add(occ_1, side_name, pv.piece_idx, seg_h['seg_id'], seg_h['length_mm'])
                occ_add(occ_1, 'left',    pv.piece_idx, seg_v['seg_id'], seg_v['length_mm'])

                nc = _build_candidates(
                    pieces_variants, {pv.piece_idx}, corners_list,
                    frame.px_per_mm, side_name, seg_h['length_mm'],
                    target, end_pos, fwd_is_horiz, _SIDE_TOLERANCE,
                )

                canvas, ox, oy = _make_canvas()
                placed_mm = _place_contour(contour_raw, seg_h, seg_v, position, frame.px_per_mm)
                _draw_piece(canvas, placed_mm, ox, oy, color, f"P{pv.piece_idx}@{position}")
                _draw_segments(canvas, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
                _mark_corner(canvas, position, ox, oy, label=f"expected {position}")

                centroid_info = f"centroid={pb.centroid_px}" if pb.centroid_px else ""
                title = (f"P{pv.piece_idx} at {position}  "
                         f"H:{seg_h['seg_id']}({seg_h['length_mm']:.1f}mm)  "
                         f"V:{seg_v['seg_id']}({seg_v['length_mm']:.1f}mm)  "
                         f"{centroid_info}")
                cv2.putText(canvas, title, (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
                _draw_legend(canvas, ox, oy, _side_lines(occ_1) + [''] + _candidate_lines(nc))

                folder = output_dir / "placements" / f"P{pv.piece_idx}" / position
                folder.mkdir(parents=True, exist_ok=True)
                fname = folder / "start.png"
                cv2.imwrite(str(fname), canvas)
                print(f"  [OUT] {fname.relative_to(output_dir)}")


def visualize_second_placements(
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
    tolerance:       float = _SIDE_TOLERANCE,
) -> None:
    """For every first-piece scenario, visualize all piece-2 candidates."""
    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        color_1   = COLORS[pv.piece_idx % len(COLORS)]
        contour_1 = corners_list[pv.piece_idx]['contour_flat']

        for v in corner_variants:
            for position in ('TL', 'BL'):
                seg_h, seg_v = (v.edges[0], v.edges[1]) if position == 'TL' else (v.edges[1], v.edges[0])
                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]
                offset_1 = seg_h['length_mm']
                folder   = output_dir / "placements" / f"P{pv.piece_idx}" / position
                folder.mkdir(parents=True, exist_ok=True)

                occ_1 = empty_occupancy()
                occ_add(occ_1, side_name, pv.piece_idx, seg_h['seg_id'], seg_h['length_mm'])
                occ_add(occ_1, 'left',    pv.piece_idx, seg_v['seg_id'], seg_v['length_mm'])

                base_1, ox, oy = _make_canvas()
                placed_1 = _place_contour(contour_1, seg_h, seg_v, position, frame.px_per_mm)
                _draw_piece(base_1, placed_1, ox, oy, color_1, f"P{pv.piece_idx}@{position}")
                _draw_segments(base_1, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
                _mark_corner(base_1, position, ox, oy)

                # ── Piece 2 ──────────────────────────────────────────────────
                for n, cand in enumerate(
                    _build_candidates(
                        pieces_variants, {pv.piece_idx}, corners_list,
                        frame.px_per_mm, side_name, offset_1,
                        target, end_pos, fwd_is_horiz, tolerance,
                    ), 1
                ):
                    occ_2 = {k: list(vals) for k, vals in occ_1.items()}
                    occ_add_candidate(occ_2, cand, fwd_side=side_name, turn_side='right')

                    nc = (
                        _build_candidates(
                            pieces_variants, {pv.piece_idx, cand.pv.piece_idx}, corners_list,
                            frame.px_per_mm, side_name, offset_1 + cand.seg_len,
                            target, end_pos, fwd_is_horiz, tolerance,
                        )
                        if (cand.valid and not cand.is_corner) else []
                    )

                    canvas = base_1.copy()
                    _draw_candidate(canvas, ox, oy, cand,
                                    end_pos, frame, side_name, offset_1, color_1, target)
                    title = (f"P{pv.piece_idx}@{position}  H={offset_1:.0f}mm"
                             f"  ->{side_name}({target:.0f}mm)  +  {cand.label}")
                    cv2.putText(canvas, title, (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                    _draw_verdict(canvas, cand.valid, cand.reason)
                    _draw_legend(canvas, ox, oy, _side_lines(occ_2) + [''] + _candidate_lines(nc))

                    tag       = "VALID" if cand.valid else "INVALID"
                    p2_folder = folder / f"P{cand.pv.piece_idx}"
                    p2_folder.mkdir(parents=True, exist_ok=True)
                    fname = p2_folder / f"{n:03d}_{cand.variant.type}_{tag}.png"
                    cv2.imwrite(str(fname), canvas)
                    print(f"  [OUT] {fname.relative_to(output_dir)}")

                    color_2 = COLORS[cand.pv.piece_idx % len(COLORS)]

                    # ── Piece 3 — scenario A: piece 2 was a valid edge ────────
                    if nc:
                        offset_3 = offset_1 + cand.seg_len

                        base_2 = base_1.copy()
                        _draw_piece(base_2, cand.placed_mm, ox, oy, color_2, cand.label)

                        for m, cand3 in enumerate(nc, 1):
                            occ_3 = {k: list(vals) for k, vals in occ_2.items()}
                            occ_add_candidate(occ_3, cand3, fwd_side=side_name, turn_side='right')

                            canvas3 = base_2.copy()
                            _draw_candidate(canvas3, ox, oy, cand3,
                                            end_pos, frame, side_name, offset_3, color_2, target)
                            title3 = (f"P{pv.piece_idx}@{position} +P{cand.pv.piece_idx}({cand.seg_len:.0f}mm)"
                                      f"  ->{side_name}({target:.0f}mm)  +  {cand3.label}")
                            cv2.putText(canvas3, title3, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                            _draw_verdict(canvas3, cand3.valid, cand3.reason)
                            _draw_legend(canvas3, ox, oy, _side_lines(occ_3))

                            tag3      = "VALID" if cand3.valid else "INVALID"
                            p3_folder = p2_folder / f"P{cand3.pv.piece_idx}"
                            p3_folder.mkdir(parents=True, exist_ok=True)
                            fname3 = p3_folder / f"{m:03d}_{cand3.variant.type}_{tag3}.png"
                            cv2.imwrite(str(fname3), canvas3)
                            print(f"  [OUT] {fname3.relative_to(output_dir)}")

                    # ── Piece 3 — scenario B: piece 2 was a valid corner ──────
                    if cand.valid and cand.is_corner and cand.seg_v is not None:
                        # piece 2 already covers part of the right side via seg_v.
                        # TL→TR→BR: fill right side top-down, starting below piece 2.
                        # BL→BR→TR: fill right side bottom-up (start_from_end=True),
                        #            starting above piece 2 and closing at TR.
                        if position == 'TL':
                            offset_b        = cand.seg_v['length_mm']
                            target_b        = PUZZLE_HEIGHT_MM
                            end_b           = 'BR'
                            turn_b          = 'bottom'
                            start_from_end_b = False
                        else:  # BL
                            offset_b        = cand.seg_v['length_mm']
                            target_b        = PUZZLE_HEIGHT_MM
                            end_b           = 'TR'
                            turn_b          = 'top'
                            start_from_end_b = True

                        nc_b = _build_candidates(
                            pieces_variants, {pv.piece_idx, cand.pv.piece_idx},
                            corners_list, frame.px_per_mm,
                            'right', offset_b, target_b, end_b,
                            False, tolerance, start_from_end_b,
                        )

                        base_b = base_1.copy()
                        _draw_piece(base_b, cand.placed_mm, ox, oy, color_2, cand.label)
                        _draw_segments(base_b, cand.seg_h, cand.seg_v, end_pos,
                                       frame.px_per_mm, ox, oy)
                        _mark_corner(base_b, end_pos, ox, oy)

                        for m, cand3 in enumerate(nc_b, 1):
                            occ_3b = {k: list(vals) for k, vals in occ_2.items()}
                            occ_add_candidate(occ_3b, cand3, fwd_side='right', turn_side=turn_b)

                            canvas3b = base_b.copy()
                            _draw_candidate(canvas3b, ox, oy, cand3,
                                            end_b, frame, 'right', offset_b, color_2, target_b)
                            title3b = (f"P{pv.piece_idx}@{position} +P{cand.pv.piece_idx}@{end_pos}"
                                       f"  ->right({target_b:.0f}mm)  +  {cand3.label}")
                            cv2.putText(canvas3b, title3b, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                            _draw_verdict(canvas3b, cand3.valid, cand3.reason)
                            _draw_legend(canvas3b, ox, oy, _side_lines(occ_3b))

                            tag3b      = "VALID" if cand3.valid else "INVALID"
                            p3b_folder = p2_folder / f"B_P{cand3.pv.piece_idx}"
                            p3b_folder.mkdir(parents=True, exist_ok=True)
                            fname3b = p3b_folder / f"{m:03d}_{cand3.variant.type}_{tag3b}.png"
                            cv2.imwrite(str(fname3b), canvas3b)
                            print(f"  [OUT] {fname3b.relative_to(output_dir)}")
