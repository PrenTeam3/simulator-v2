"""Placement debug — visual check of TL and BL starting positions."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_solverv2.frame import PuzzleFrame
from puzzle_solverv2.border_info import PieceBorderInfo
from puzzle_solverv2.variants import PieceVariants
from puzzle_solverv2._placement_geometry import (
    _PUZZLE_CORNERS_MM,
    _SECOND_CONFIGS,
    _SIDE_TOLERANCE,
    _place_contour,
    _place_contour_on_side,
)
from puzzle_solverv2._placement_drawing import (
    COLORS,
    _to_px,
    _make_canvas,
    _draw_piece,
    _draw_segments,
    _draw_verdict,
    _draw_progress_bar,
)


def visualize_start_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
) -> None:
    """
    For each piece with a corner variant, save two images:
      dbg_place_P{idx}_TL.png
      dbg_place_P{idx}_BL.png

    Each image shows the piece placed at TL or BL within the padded frame.
    Horiz segment = cyan, vert segment = orange.
    """
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
                # TL: edges[0] → horiz (along top, going right),  edges[1] → vert (↓ along left)
                # BL: edges[1] → horiz (along bottom, going right),edges[0] → vert (↑ along left)
                # Swapping gives the 90° rotation needed to go from TL to BL
                seg_h, seg_v = (v.edges[0], v.edges[1]) if position == 'TL' else (v.edges[1], v.edges[0])

                canvas, ox, oy = _make_canvas()
                placed_mm = _place_contour(contour_raw, seg_h, seg_v, position, frame.px_per_mm)
                label = f"P{pv.piece_idx}@{position}"
                _draw_piece(canvas, placed_mm, ox, oy, color, label)
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

                fname = output_dir / f"dbg_place_P{pv.piece_idx}_{position}.png"
                cv2.imwrite(str(fname), canvas)
                print(f"  [OUT] {fname.name}")


def visualize_second_placements(
    pieces_border:   list[PieceBorderInfo],
    pieces_variants: list[PieceVariants],
    corners_list:    list[dict],
    frame:           PuzzleFrame,
    output_dir:      Path,
    tolerance:       float = _SIDE_TOLERANCE,
) -> None:
    """
    For every first-piece scenario (corner piece at TL or BL) and every possible
    second piece, save one image clearly labelled VALID or INVALID.

    Validity rules:
      - Edge piece:   new_total < target - tolerance  (room left for closing corner)
      - Corner piece: |new_total - target| <= tolerance  (closes the side within ±tol)

    Files: dbg_step2_P{p1}_{pos}_{n:03d}_P{p2}_{type}_{VALID|INVALID}.png
    """
    for pv in pieces_variants:
        corner_variants = [v for v in pv.variants if v.type == 'corner']
        if not corner_variants:
            continue

        contour_1 = corners_list[pv.piece_idx]['contour_flat']
        color_1   = COLORS[pv.piece_idx % len(COLORS)]

        for v in corner_variants:
            for position in ('TL', 'BL'):
                seg_h, seg_v = (v.edges[0], v.edges[1]) if position == 'TL' else (v.edges[1], v.edges[0])

                side_name, target, end_pos, fwd_is_horiz = _SECOND_CONFIGS[position]
                current_length = seg_h['length_mm']
                n = 0   # image counter per scenario

                base_canvas, ox, oy = _make_canvas()
                placed_1 = _place_contour(contour_1, seg_h, seg_v, position, frame.px_per_mm)
                _draw_piece(base_canvas, placed_1, ox, oy, color_1, f"P{pv.piece_idx}@{position}")
                _draw_segments(base_canvas, seg_h, seg_v, position, frame.px_per_mm, ox, oy)
                fx, fy = _to_px(_PUZZLE_CORNERS_MM[position], ox, oy)
                cv2.drawMarker(base_canvas, (fx, fy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                for pv2 in pieces_variants:
                    if pv2.piece_idx == pv.piece_idx:
                        continue

                    contour_2 = corners_list[pv2.piece_idx]['contour_flat']
                    color_2   = COLORS[pv2.piece_idx % len(COLORS)]

                    for v2 in pv2.variants:
                        candidates_for_v2 = []   # list of (placed_2, seg2_len, valid, reason, label, seg_h2, seg_v2, is_corner)

                        if v2.type == 'edge':
                            seg2    = v2.edges[0]
                            new_len = current_length + seg2['length_mm']
                            placed_2 = _place_contour_on_side(
                                contour_2, seg2, side_name, current_length, frame.px_per_mm
                            )
                            if new_len < target - tolerance:
                                valid  = True
                                reason = f"{current_length:.0f}+{seg2['length_mm']:.0f}={new_len:.0f}mm  (room={target-new_len:.0f}mm)"
                            else:
                                valid  = False
                                reason = f"{current_length:.0f}+{seg2['length_mm']:.0f}={new_len:.0f}mm  overshoot (tol={tolerance:.0f}mm)"
                            candidates_for_v2.append(
                                (placed_2, seg2['length_mm'], valid, reason,
                                 f"P{pv2.piece_idx} edge {seg2['seg_id']}({seg2['length_mm']:.0f}mm)",
                                 None, None, False)
                            )

                        elif v2.type == 'corner' and len(v2.edges) >= 2:
                            for ori_n, (fwd, turn) in enumerate(
                                [(v2.edges[0], v2.edges[1]), (v2.edges[1], v2.edges[0])]
                            ):
                                new_len = current_length + fwd['length_mm']
                                diff    = abs(new_len - target)
                                seg_h2  = fwd  if fwd_is_horiz else turn
                                seg_v2  = turn if fwd_is_horiz else fwd
                                placed_2 = _place_contour(
                                    contour_2, seg_h2, seg_v2, end_pos, frame.px_per_mm
                                )
                                if diff <= tolerance:
                                    valid  = True
                                    reason = (f"{current_length:.0f}+{fwd['length_mm']:.0f}"
                                              f"={new_len:.0f}mm  diff={diff:.1f}mm ≤ {tolerance:.0f}mm")
                                else:
                                    valid  = False
                                    reason = (f"{current_length:.0f}+{fwd['length_mm']:.0f}"
                                              f"={new_len:.0f}mm  diff={diff:.1f}mm > {tolerance:.0f}mm")
                                candidates_for_v2.append(
                                    (placed_2, fwd['length_mm'], valid, reason,
                                     f"P{pv2.piece_idx}@{end_pos} ori{ori_n} fwd={fwd['seg_id']}({fwd['length_mm']:.0f}mm)",
                                     seg_h2, seg_v2, True)
                                )

                        for placed_2, seg2_len, valid, reason, label, sh2, sv2, is_corner in candidates_for_v2:
                            n += 1
                            canvas = base_canvas.copy()

                            _draw_piece(canvas, placed_2, ox, oy, color_2, f"P{pv2.piece_idx}")
                            if is_corner and sh2 and sv2:
                                _draw_segments(canvas, sh2, sv2, end_pos, frame.px_per_mm, ox, oy)
                                ex, ey = _to_px(_PUZZLE_CORNERS_MM[end_pos], ox, oy)
                                cv2.drawMarker(canvas, (ex, ey), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

                            _draw_progress_bar(canvas, ox, oy, side_name,
                                               current_length, seg2_len,
                                               color_1, color_2, target)

                            title = (f"P{pv.piece_idx}@{position}  H={current_length:.0f}mm"
                                     f"  →{side_name}({target:.0f}mm)  +  {label}")
                            cv2.putText(canvas, title, (10, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

                            _draw_verdict(canvas, valid, reason)

                            tag   = "VALID" if valid else "INVALID"
                            fname = (output_dir /
                                     f"dbg_step2_P{pv.piece_idx}_{position}"
                                     f"_{n:03d}_P{pv2.piece_idx}_{v2.type}_{tag}.png")
                            cv2.imwrite(str(fname), canvas)
                            print(f"  [OUT] {fname.name}")
