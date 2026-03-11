"""Placement debug — visual check of TL and BL starting positions."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_solverv2.frame import PuzzleFrame, PUZZLE_WIDTH_MM, PUZZLE_HEIGHT_MM
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants
from puzzle_solverv2._placement_geometry import (
    _PUZZLE_CORNERS_MM,
    _SECOND_CONFIGS,
    _SIDE_TOLERANCE,
    _place_contour,
    _build_candidates,
)
from puzzle_solverv2._placement_drawing import (
    COLORS,
    _to_px,
    _make_canvas,
    _draw_piece,
    _draw_segments,
    _draw_verdict,
    _draw_progress_bar,
    _draw_legend,
    _side_lines,
    _candidate_lines,
)

# Maps (current_side, closing_corner) → side the turn segment goes along
_CORNER_TURN_SIDE = {
    ('top',    'TR'): 'right',
    ('bottom', 'BR'): 'right',
    ('right',  'BR'): 'bottom',
    ('right',  'TR'): 'top',
}


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

                occ_1 = {'top': [], 'right': [], 'bottom': [], 'left': []}
                occ_1[side_name].append({'piece_idx': pv.piece_idx, 'seg_id': seg_h['seg_id'], 'length_mm': seg_h['length_mm']})
                occ_1['left'].append({'piece_idx': pv.piece_idx, 'seg_id': seg_v['seg_id'], 'length_mm': seg_v['length_mm']})

                nc = _build_candidates(pieces_variants, {pv.piece_idx}, corners_list,
                                       frame.px_per_mm, side_name, seg_h['length_mm'],
                                       target, end_pos, fwd_is_horiz, _SIDE_TOLERANCE)

                canvas, ox, oy = _make_canvas()
                placed_mm = _place_contour(contour_raw, seg_h, seg_v, position, frame.px_per_mm)
                _draw_piece(canvas, placed_mm, ox, oy, color, f"P{pv.piece_idx}@{position}")
                _draw_segments(canvas, seg_h, seg_v, position, frame.px_per_mm, ox, oy)

                fx, fy = _to_px(_PUZZLE_CORNERS_MM[position], ox, oy)
                cv2.drawMarker(canvas, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(canvas, f"expected {position}", (fx + 8, fy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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


def _draw_candidate(canvas, ox, oy, placed, color, label, sh, sv, end_pos, frame,
                    side_name, bar_offset, seg_len, bar_color_base, target):
    """Draw a candidate piece + segment markers + progress bar onto canvas."""
    _draw_piece(canvas, placed, ox, oy, color, label)
    if sh and sv:
        _draw_segments(canvas, sh, sv, end_pos, frame.px_per_mm, ox, oy)
        ex, ey = _to_px(_PUZZLE_CORNERS_MM[end_pos], ox, oy)
        cv2.drawMarker(canvas, (ex, ey), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
    _draw_progress_bar(canvas, ox, oy, side_name, bar_offset, seg_len, bar_color_base, color, target)


def visualize_second_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
    tolerance:       float = _SIDE_TOLERANCE,
) -> None:
    """For every first-piece scenario, visualize all piece-2 and piece-3 candidates."""
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

                occ_1 = {'top': [], 'right': [], 'bottom': [], 'left': []}
                occ_1[side_name].append({'piece_idx': pv.piece_idx, 'seg_id': seg_h['seg_id'], 'length_mm': seg_h['length_mm']})
                occ_1['left'].append({'piece_idx': pv.piece_idx, 'seg_id': seg_v['seg_id'], 'length_mm': seg_v['length_mm']})

                base_1, ox, oy = _make_canvas()
                placed_1 = _place_contour(contour_1, seg_h, seg_v, position, frame.px_per_mm)
                _draw_piece(base_1, placed_1, ox, oy, color_1, f"P{pv.piece_idx}@{position}")
                _draw_segments(base_1, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
                fx, fy = _to_px(_PUZZLE_CORNERS_MM[position], ox, oy)
                cv2.drawMarker(base_1, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                # ── Piece 2 ──────────────────────────────────────────────────
                for n, (pv2, v2, placed_2, seg2_len, valid, reason, label, sh2, sv2, is_corner) in enumerate(
                    _build_candidates(pieces_variants, {pv.piece_idx}, corners_list,
                                      frame.px_per_mm, side_name, offset_1,
                                      target, end_pos, fwd_is_horiz, tolerance), 1):
                    color_2 = COLORS[pv2.piece_idx % len(COLORS)]

                    occ_2 = {k: list(vals) for k, vals in occ_1.items()}
                    if is_corner:
                        occ_2[side_name].append({'piece_idx': pv2.piece_idx, 'seg_id': sh2['seg_id'], 'length_mm': seg2_len})
                        occ_2['right'].append({'piece_idx': pv2.piece_idx, 'seg_id': sv2['seg_id'], 'length_mm': sv2['length_mm']})
                    else:
                        occ_2[side_name].append({'piece_idx': pv2.piece_idx, 'seg_id': v2.edges[0]['seg_id'], 'length_mm': seg2_len})

                    # Pre-compute piece-3 candidates (reused in loop + shown in legend)
                    if valid and not is_corner:
                        side_3, offset_3, target_3, end_3, fwd_3 = (
                            side_name, offset_1 + seg2_len, target, end_pos, fwd_is_horiz)
                        nc = _build_candidates(pieces_variants, {pv.piece_idx, pv2.piece_idx}, corners_list,
                                               frame.px_per_mm, side_3, offset_3, target_3, end_3, fwd_3, tolerance)
                    elif valid and is_corner:
                        turn_len = sv2['length_mm']
                        if end_pos == 'TR':
                            side_3, offset_3, target_3, end_3 = 'right', turn_len, PUZZLE_HEIGHT_MM, 'BR'
                        else:
                            side_3, offset_3, target_3, end_3 = 'right', 0.0, PUZZLE_HEIGHT_MM - turn_len, 'TR'
                        fwd_3 = False
                        print(f"  [DBG] P{pv.piece_idx}@{position}+P{pv2.piece_idx} corner→right: side={side_3} offset={offset_3:.1f} target={target_3:.1f} end={end_3}")
                        nc = _build_candidates(pieces_variants, {pv.piece_idx, pv2.piece_idx}, corners_list,
                                               frame.px_per_mm, side_3, offset_3, target_3, end_3, fwd_3, tolerance)
                        print(f"  [DBG] → {len(nc)} candidates: {[(r[0].piece_idx, r[1].type, f'{r[3]:.0f}mm', 'V' if r[4] else 'X') for r in nc]}")
                    else:
                        nc = []

                    canvas = base_1.copy()
                    _draw_candidate(canvas, ox, oy, placed_2, color_2, f"P{pv2.piece_idx}",
                                    sh2, sv2, end_pos, frame, side_name, offset_1, seg2_len, color_1, target)
                    title = (f"P{pv.piece_idx}@{position}  H={offset_1:.0f}mm"
                             f"  →{side_name}({target:.0f}mm)  +  {label}")
                    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                    _draw_verdict(canvas, valid, reason)
                    _draw_legend(canvas, ox, oy, _side_lines(occ_2) + [''] + _candidate_lines(nc))
                    tag       = "VALID" if valid else "INVALID"
                    p2_folder = folder / f"P{pv2.piece_idx}"
                    p2_folder.mkdir(parents=True, exist_ok=True)
                    fname = p2_folder / f"{n:03d}_{v2.type}_{tag}.png"
                    cv2.imwrite(str(fname), canvas)
                    print(f"  [OUT] {fname.relative_to(output_dir)}")

                    if not nc:
                        continue

                    # ── Piece 3 ───────────────────────────────────────────────
                    base_2 = base_1.copy()
                    _draw_piece(base_2, placed_2, ox, oy, color_2, f"P{pv2.piece_idx}")
                    if is_corner and sh2 and sv2:
                        _draw_segments(base_2, sh2, sv2, end_pos, frame.px_per_mm, ox, oy)
                        ex, ey = _to_px(_PUZZLE_CORNERS_MM[end_pos], ox, oy)
                        cv2.drawMarker(base_2, (ex, ey), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                    for m, (pv3, v3, placed_3, seg3_len, valid3, reason3, label3, sh3, sv3, is_corner3) in enumerate(nc, 1):
                        color_3 = COLORS[pv3.piece_idx % len(COLORS)]

                        occ_3 = {k: list(vals) for k, vals in occ_2.items()}
                        if is_corner3 and sh3 and sv3:
                            occ_3[side_3].append({'piece_idx': pv3.piece_idx, 'seg_id': sh3['seg_id'], 'length_mm': seg3_len})
                            turn_side = _CORNER_TURN_SIDE.get((side_3, end_3))
                            if turn_side:
                                occ_3[turn_side].append({'piece_idx': pv3.piece_idx, 'seg_id': sv3['seg_id'], 'length_mm': sv3['length_mm']})
                        else:
                            occ_3[side_3].append({'piece_idx': pv3.piece_idx, 'seg_id': v3.edges[0]['seg_id'], 'length_mm': seg3_len})

                        canvas3 = base_2.copy()
                        _draw_candidate(canvas3, ox, oy, placed_3, color_3, f"P{pv3.piece_idx}",
                                        sh3, sv3, end_3, frame, side_3, offset_3, seg3_len, color_2, target_3)
                        title3 = (f"P{pv.piece_idx}@{position} +P{pv2.piece_idx}({seg2_len:.0f}mm)"
                                  f"  →{side_3}({target_3:.0f}mm)  +  {label3}")
                        cv2.putText(canvas3, title3, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
                        _draw_verdict(canvas3, valid3, reason3)
                        _draw_legend(canvas3, ox, oy, _side_lines(occ_3))
                        tag3      = "VALID" if valid3 else "INVALID"
                        p3_folder = p2_folder / f"P{pv3.piece_idx}"
                        p3_folder.mkdir(parents=True, exist_ok=True)
                        fname3 = p3_folder / f"{m:03d}_{v3.type}_{tag3}.png"
                        cv2.imwrite(str(fname3), canvas3)
                        print(f"  [OUT] {fname3.relative_to(output_dir)}")
