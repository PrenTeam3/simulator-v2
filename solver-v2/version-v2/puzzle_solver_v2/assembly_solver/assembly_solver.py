"""Assembly solver for placing puzzle pieces together based on edge solver results."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ..common.common import SolverUtils


@dataclass
class PlacedPiece:
    """Represents a piece that has been placed in the assembly."""
    piece_id: int
    position: np.ndarray  # (x, y) position
    rotation: float  # rotation angle in degrees
    contour_points: np.ndarray  # transformed contour points


@dataclass
class AssemblyStep:
    """Represents one step in the assembly process for visualization."""
    step_number: int
    step_name: str
    description: str
    visualization_data: Dict  # Data needed to visualize this step


class AssemblySolver:
    """Handles the assembly phase: placing puzzle pieces together."""

    @staticmethod
    def assemble_puzzle(edge_solver_results, prepared_data, show_visualizations=True):
        """Assemble puzzle pieces using edge solver results.

        Args:
            edge_solver_results: Results from EdgeSolver.solve_with_edges()
            prepared_data: Dictionary from PuzzlePreparer.prepare_puzzle_data()
            show_visualizations: Whether to display visualization windows

        Returns:
            dict: Assembly results containing placed pieces and visualizations
        """
        SolverUtils.print_section_header("PUZZLE ASSEMBLY PHASE")

        # Extract data from edge solver results
        all_pieces = edge_solver_results['all_pieces']
        piece_frame_segments = edge_solver_results['piece_frame_segments']
        best_chains = edge_solver_results['best_chains']

        # Extract prepared data
        original_image = prepared_data['original_image']
        project_root = prepared_data['project_root']

        # Build piece connections from best chains
        from ..edge_solver_v2.connection_selector import ConnectionSelector
        piece_connections = ConnectionSelector.determine_best_connections(
            best_chains,
            piece_frame_segments
        )

        print(f"\nStarting assembly with {len(all_pieces)} pieces")

        # Storage for assembly steps
        assembly_steps = []

        # STEP 1: Calculate/verify centroids and red/blue dots for all pieces
        print("\n" + "-"*70)
        print("Step 1: Calculating centroids and chain endpoints for all pieces...")
        print("-"*70)

        piece_centroids = {}
        piece_chain_endpoints = {}  # Store red/blue dots for each piece's chains

        for piece in all_pieces:
            # Centroid is already calculated in the piece data
            centroid = piece.centroid
            piece_centroids[piece.piece_id] = np.array([centroid.x, centroid.y])
            print(f"  Piece {piece.piece_id}: centroid at ({centroid.x:.1f}, {centroid.y:.1f})")

        # Get red/blue dots from chain matching results (already calculated correctly!)
        all_segments = prepared_data['all_segments']

        for piece_id, connections in piece_connections.items():
            piece = next(p for p in all_pieces if p.piece_id == piece_id)
            segments = all_segments[piece_id]

            piece_chain_endpoints[piece_id] = []

            for connected_piece_id, chain, side in connections:
                # Determine which segments belong to this piece in the chain
                if chain.piece1_id == piece_id:
                    seg_ids = chain.segment_ids_p1
                    # Use pre-calculated blue/red dots from chain matching
                    B = chain.blue_dot_p1.copy() if chain.blue_dot_p1 is not None else None
                    R = chain.red_dot_p1.copy() if chain.red_dot_p1 is not None else None
                else:
                    seg_ids = chain.segment_ids_p2
                    # Use pre-calculated blue/red dots from chain matching
                    B = chain.blue_dot_p2.copy() if chain.blue_dot_p2 is not None else None
                    R = chain.red_dot_p2.copy() if chain.red_dot_p2 is not None else None

                # Get chain points (ORIGINAL, no transformations) for later use
                chain_points = []
                for seg_id in seg_ids:
                    seg = segments[seg_id]
                    for pt in seg.contour_points:
                        chain_points.append([pt.x, pt.y])
                chain_points = np.array(chain_points)

                if B is not None and R is not None:
                    piece_chain_endpoints[piece_id].append({
                        'connected_to': connected_piece_id,
                        'chain': chain,
                        'blue_dot': B,
                        'red_dot': R,
                        'chain_points': chain_points
                    })

                    print(f"    Chain to P{connected_piece_id}: Blue=({B[0]:.1f},{B[1]:.1f}), Red=({R[0]:.1f},{R[1]:.1f})")

        # Store step 1 data
        step1_data = {
            'pieces': all_pieces,
            'centroids': piece_centroids,
            'chain_endpoints': piece_chain_endpoints,
            'original_image': original_image
        }
        assembly_steps.append(AssemblyStep(
            step_number=1,
            step_name="Calculate Centroids & Chain Endpoints",
            description="Find the centroid and red/blue dots for each piece",
            visualization_data=step1_data
        ))

        print("\n[OK] Step 1 complete - centroids and chain endpoints calculated for all pieces")

        # STEP 2: Normalize piece orientations (align frame edges with axes)
        print("\n" + "-"*70)
        print("Step 2: Normalizing piece orientations (axis alignment)...")
        print("-"*70)

        piece_orientations = {}
        normalized_pieces = {}

        for piece in all_pieces:
            # Find frame corners to determine orientation
            if len(piece.frame_corners) == 0:
                print(f"  [WARNING] Piece {piece.piece_id} has no frame corners, skipping orientation")
                piece_orientations[piece.piece_id] = 0.0
                continue

            # Get frame corner position
            frame_corner = piece.frame_corners[0]
            frame_pos = np.array([frame_corner.x, frame_corner.y])

            # Find the two segments adjacent to the frame corner
            # These should be the straight edges
            frame_adjacent_segments = []
            for segment in piece.segments:
                seg_start = np.array([segment.corner_start.x, segment.corner_start.y])
                seg_end = np.array([segment.corner_end.x, segment.corner_end.y])

                # Check if segment is adjacent to frame corner
                dist_to_start = np.linalg.norm(seg_start - frame_pos)
                dist_to_end = np.linalg.norm(seg_end - frame_pos)

                if dist_to_start < 1.0 or dist_to_end < 1.0:  # Very close to frame corner
                    if segment.is_straight:
                        frame_adjacent_segments.append(segment)

            if len(frame_adjacent_segments) < 2:
                print(f"  [WARNING] Piece {piece.piece_id} has only {len(frame_adjacent_segments)} straight frame edges")
                piece_orientations[piece.piece_id] = 0.0
                continue

            # Calculate angles of the two straight edges
            angles = []
            for segment in frame_adjacent_segments[:2]:
                seg_start = np.array([segment.corner_start.x, segment.corner_start.y])
                seg_end = np.array([segment.corner_end.x, segment.corner_end.y])

                # Vector from start to end
                vec = seg_end - seg_start
                angle = np.arctan2(vec[1], vec[0])  # Angle in radians
                angles.append(np.degrees(angle))

            # The two straight edges should be perpendicular
            # STEP 1: Align edges to X/Y axes (minimal rotation)
            angle1, angle2 = angles[0], angles[1]

            # Normalize angles to [-180, 180]
            angle1 = (angle1 + 180) % 360 - 180
            angle2 = (angle2 + 180) % 360 - 180

            # Find which angle is closest to 0° or 90°
            # We want to rotate to align with 0° (horizontal) or 90° (vertical)
            candidates = [0, 90, 180, -90]

            # Calculate rotation needed for angle1 to reach each candidate
            rotations1 = [(candidate - angle1) for candidate in candidates]
            rotations2 = [(candidate - angle2) for candidate in candidates]

            # Choose the rotation that's closest to 0 (minimal rotation)
            all_rotations = rotations1 + rotations2
            axis_alignment_rotation = min(all_rotations, key=lambda x: abs(x))

            # Normalize to [-180, 180]
            axis_alignment_rotation = (axis_alignment_rotation + 180) % 360 - 180

            # STEP 2: After axis alignment, determine which way the edges point
            # and add 0°, 90°, 180°, or 270° to orient them correctly
            aligned_angle1 = (angle1 + axis_alignment_rotation + 180) % 360 - 180
            aligned_angle2 = (angle2 + axis_alignment_rotation + 180) % 360 - 180

            # Now the edges are aligned with axes. Check which direction they point.
            # We want: one edge at 0° (+X, right) and one at -90° (-Y, down)
            # Try adding 0°, 90°, 180°, 270° and see which gives us the right orientation

            best_additional_rotation = 0
            best_score = float('inf')

            for additional_rotation in [0, 90, 180, 270]:
                final_angle1 = (aligned_angle1 + additional_rotation + 180) % 360 - 180
                final_angle2 = (aligned_angle2 + additional_rotation + 180) % 360 - 180

                # Check if we have one edge at ~0° and one at ~-90°
                # Calculate minimum distance to target
                has_0_deg = (abs(final_angle1) < 15 or abs(final_angle2) < 15 or
                            abs(final_angle1 - 180) < 15 or abs(final_angle2 - 180) < 15)
                has_minus90_deg = (abs(final_angle1 - (-90)) < 15 or abs(final_angle2 - (-90)) < 15 or
                                  abs(final_angle1 - 90) < 15 or abs(final_angle2 - 90) < 15)

                # Calculate score: prefer configuration with one edge at 0° and one at -90°
                dist_to_0 = min(abs(final_angle1), abs(final_angle2), abs(final_angle1 - 180), abs(final_angle2 - 180))
                dist_to_minus90 = min(abs(final_angle1 + 90), abs(final_angle2 + 90), abs(final_angle1 - 90), abs(final_angle2 - 90))

                score = dist_to_0 + dist_to_minus90

                if score < best_score:
                    best_score = score
                    best_additional_rotation = additional_rotation

            # Total rotation = axis alignment + additional rotation to correct direction
            best_rotation = axis_alignment_rotation + best_additional_rotation

            # Normalize to [-180, 180]
            best_rotation = (best_rotation + 180) % 360 - 180

            piece_orientations[piece.piece_id] = best_rotation
            print(f"  Piece {piece.piece_id}: rotation = {best_rotation:.1f}° (frame edges at {angle1:.1f}° and {angle2:.1f}°)")

            # Apply rotation to piece contour points around centroid
            centroid = piece_centroids[piece.piece_id]
            contour_points = np.array([[p.x, p.y] for p in piece.contour_points])

            # Translate to origin
            translated = contour_points - centroid

            # Rotate
            theta = np.radians(best_rotation)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated = translated @ rotation_matrix.T

            # Translate back
            normalized_contour = rotated + centroid

            normalized_pieces[piece.piece_id] = {
                'piece': piece,
                'rotation': best_rotation,
                'normalized_contour': normalized_contour,
                'original_contour': contour_points,
                'centroid': centroid
            }

        # Store step 2 data
        step2_data = {
            'pieces': all_pieces,
            'piece_orientations': piece_orientations,
            'normalized_pieces': normalized_pieces,
            'centroids': piece_centroids,
            'original_image': original_image
        }
        assembly_steps.append(AssemblyStep(
            step_number=2,
            step_name="Normalize Orientations",
            description="Rotate pieces to align frame edges with x and y axes",
            visualization_data=step2_data
        ))

        print("\n[OK] Step 2 complete - all pieces normalized to axis alignment")

        # STEP 3: Find the best anchor piece (corner with highest connections)
        print("\n" + "-"*70)
        print("Step 3: Finding the best anchor piece...")
        print("-"*70)

        # Calculate connection scores for each piece
        piece_scores = {}
        for piece_id, connections in piece_connections.items():
            # Sum the shape scores of the two connections
            total_score = 0
            connection_details = []

            for connected_piece_id, chain, side in connections:
                score = chain.shape_score
                total_score += score
                connection_details.append({
                    'connected_to': connected_piece_id,
                    'score': score,
                    'side': side,
                    'chain_length': chain.chain_length
                })

            piece_scores[piece_id] = {
                'total_score': total_score,
                'avg_score': total_score / len(connections) if connections else 0,
                'num_connections': len(connections),
                'connections': connection_details
            }

            print(f"  Piece {piece_id}: {len(connections)} connections, total score = {total_score:.1f}")
            for detail in connection_details:
                print(f"    -> P{detail['connected_to']} (side {detail['side']}): {detail['score']:.1f}% (chain len: {detail['chain_length']})")

        # Find the piece with the highest total connection score
        # This should be a corner frame piece with strong connections
        if piece_scores:
            anchor_piece_id = max(piece_scores.keys(), key=lambda pid: piece_scores[pid]['total_score'])
            anchor_score = piece_scores[anchor_piece_id]['total_score']

            print(f"\n[SELECTED] Anchor piece: P{anchor_piece_id} with total score {anchor_score:.1f}")
            print(f"  This piece has the strongest connections and will be placed first")
        else:
            print("\n[ERROR] No pieces with connections found!")
            anchor_piece_id = 0
            anchor_score = 0

        # Store step 3 data
        step3_data = {
            'pieces': all_pieces,
            'normalized_pieces': normalized_pieces,
            'piece_scores': piece_scores,
            'anchor_piece_id': anchor_piece_id,
            'anchor_score': anchor_score,
            'piece_connections': piece_connections,
            'original_image': original_image
        }
        assembly_steps.append(AssemblyStep(
            step_number=3,
            step_name="Select Anchor Piece",
            description="Choose the corner piece with the highest connection scores",
            visualization_data=step3_data
        ))

        print("\n[OK] Step 3 complete - anchor piece selected")

        # STEP 4: Position anchor piece at origin (top-left)
        print("\n" + "-"*70)
        print("Step 4: Positioning anchor piece at origin...")
        print("-"*70)

        placed_pieces = {}

        # Get the anchor piece
        anchor_piece = next(p for p in all_pieces if p.piece_id == anchor_piece_id)
        anchor_normalized = normalized_pieces[anchor_piece_id]

        # Get the frame corner of the anchor piece
        if len(anchor_piece.frame_corners) > 0:
            frame_corner = anchor_piece.frame_corners[0]
            frame_corner_pos = np.array([frame_corner.x, frame_corner.y])

            # Apply the rotation to the frame corner position
            centroid = piece_centroids[anchor_piece_id]
            rotation_angle = piece_orientations[anchor_piece_id]

            # Rotate frame corner around centroid
            translated = frame_corner_pos - centroid
            theta = np.radians(rotation_angle)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            rotated_corner = rotation_matrix @ translated
            rotated_corner_pos = rotated_corner + centroid

            # Define target position (top-left corner with some margin)
            margin = 100
            target_position = np.array([margin, margin])

            # Calculate translation needed to move rotated frame corner to target
            translation = target_position - rotated_corner_pos

            print(f"  Anchor piece: P{anchor_piece_id}")
            print(f"  Frame corner at: ({rotated_corner_pos[0]:.1f}, {rotated_corner_pos[1]:.1f})")
            print(f"  Target position: ({target_position[0]:.1f}, {target_position[1]:.1f})")
            print(f"  Translation: ({translation[0]:.1f}, {translation[1]:.1f})")

            # Apply translation to the normalized contour
            final_contour = anchor_normalized['normalized_contour'] + translation
            final_centroid = centroid + translation

            # CRITICAL: Ensure the frame corner is at the TOP-LEFT of the piece
            # The frame corner should be at the minimum X and minimum Y of the contour
            # This ensures straight edges align with image frame (X+ right, Y+ down)

            print(f"  Checking if frame corner is at top-left...")

            # Try rotations: 0°, 90°, 180°, 270° around the centroid
            best_additional_rotation = 0
            best_score = float('inf')

            for additional_rotation in [0, 90, 180, 270]:
                # Apply additional rotation to the contour around the centroid
                theta_add = np.radians(additional_rotation)
                rotation_matrix_add = np.array([
                    [np.cos(theta_add), -np.sin(theta_add)],
                    [np.sin(theta_add), np.cos(theta_add)]
                ])

                # Rotate the contour
                test_contour = rotation_matrix_add @ (final_contour - final_centroid).T
                test_contour = test_contour.T + final_centroid

                # Rotate the frame corner position
                frame_corner_placed = rotated_corner_pos + translation
                test_frame_corner = rotation_matrix_add @ (frame_corner_placed - final_centroid) + final_centroid

                # Calculate bounding box
                min_x = np.min(test_contour[:, 0])
                min_y = np.min(test_contour[:, 1])

                # Calculate distance from frame corner to top-left (min_x, min_y)
                distance_to_top_left = abs(test_frame_corner[0] - min_x) + abs(test_frame_corner[1] - min_y)

                print(f"    Rotation {additional_rotation} deg: frame corner at ({test_frame_corner[0]:.1f}, {test_frame_corner[1]:.1f}), top-left at ({min_x:.1f}, {min_y:.1f}), distance = {distance_to_top_left:.1f}")

                if distance_to_top_left < best_score:
                    best_score = distance_to_top_left
                    best_additional_rotation = additional_rotation

            print(f"  [OK] Best orientation: +{best_additional_rotation} deg rotation (distance to top-left: {best_score:.1f})")

            # Apply the rotation (even if 0°, we still need to recalculate positions)
            theta_final = np.radians(best_additional_rotation)
            rotation_matrix_final = np.array([
                [np.cos(theta_final), -np.sin(theta_final)],
                [np.sin(theta_final), np.cos(theta_final)]
            ])

            # Rotate contour around centroid
            final_contour = rotation_matrix_final @ (final_contour - final_centroid).T
            final_contour = final_contour.T + final_centroid

            # Rotate the frame corner position
            frame_corner_placed = rotated_corner_pos + translation
            final_frame_corner = rotation_matrix_final @ (frame_corner_placed - final_centroid) + final_centroid

            # Now translate so the frame corner is at (100, 100)
            target_position_final = np.array([100, 100])
            final_translation = target_position_final - final_frame_corner

            final_contour = final_contour + final_translation
            final_centroid = final_centroid + final_translation

            print(f"  Frame corner repositioned to: ({target_position_final[0]:.1f}, {target_position_final[1]:.1f})")

            # Update total rotation
            rotation_angle = rotation_angle + best_additional_rotation

            # Store the placed anchor piece
            placed_pieces[anchor_piece_id] = PlacedPiece(
                piece_id=anchor_piece_id,
                position=final_centroid,
                rotation=rotation_angle,
                contour_points=final_contour
            )

            print(f"  Final centroid: ({final_centroid[0]:.1f}, {final_centroid[1]:.1f})")
            print(f"  Final rotation: {rotation_angle:.1f}°")
        else:
            print(f"  [WARNING] Anchor piece has no frame corner, placing at default position")
            margin = 100
            target_centroid = np.array([margin + 200, margin + 200])
            translation = target_centroid - piece_centroids[anchor_piece_id]
            final_contour = anchor_normalized['normalized_contour'] + translation

            placed_pieces[anchor_piece_id] = PlacedPiece(
                piece_id=anchor_piece_id,
                position=target_centroid,
                rotation=piece_orientations[anchor_piece_id],
                contour_points=final_contour
            )

        # Store step 4 data (make a copy of placed_pieces so Step 5 doesn't modify it)
        step4_data = {
            'pieces': all_pieces,
            'normalized_pieces': normalized_pieces,
            'anchor_piece_id': anchor_piece_id,
            'placed_pieces': placed_pieces.copy(),  # IMPORTANT: Make a copy!
            'original_image': original_image
        }
        assembly_steps.append(AssemblyStep(
            step_number=4,
            step_name="Position Anchor",
            description="Place anchor piece at top-left corner",
            visualization_data=step4_data
        ))

        print("\n[OK] Step 4 complete - anchor piece positioned at origin")

        # STEP 5: Place second piece (highest chain score with anchor)
        print("\n" + "-"*70)
        print("Step 5: Placing second piece (iterative placement)...")
        print("-"*70)

        # Get connections for the anchor piece
        anchor_connections = piece_connections.get(anchor_piece_id, [])

        if len(anchor_connections) > 0:
            # Find the connection with the highest score
            best_connection = max(anchor_connections, key=lambda conn: conn[1].shape_score)
            next_piece_id, best_chain, side = best_connection

            print(f"\n  Placing next piece: P{next_piece_id}")
            print(f"  Connected to anchor P{anchor_piece_id} via chain:")
            print(f"    Chain length: {best_chain.chain_length} segments")
            print(f"    Shape score: {best_chain.shape_score:.1f}%")
            print(f"    Segments: P{best_chain.piece1_id}{best_chain.segment_ids_p1} <-> P{best_chain.piece2_id}{best_chain.segment_ids_p2}")

            # Get the segment data for alignment
            all_segments = prepared_data['all_segments']

            # Get the anchor piece's placed position
            anchor_placed = placed_pieces[anchor_piece_id]

            # Determine which piece is anchor and which is next in the chain
            if best_chain.piece1_id == anchor_piece_id:
                anchor_seg_ids = best_chain.segment_ids_p1
                next_seg_ids = best_chain.segment_ids_p2
            else:
                anchor_seg_ids = best_chain.segment_ids_p2
                next_seg_ids = best_chain.segment_ids_p1

            # Get the segments for both pieces
            anchor_segments = all_segments[anchor_piece_id]
            next_segments = all_segments[next_piece_id]

            # =================================================================
            # STEP 1: Get anchor chain red/blue dots (from pre-calculated values)
            # =================================================================
            print(f"\n  Step 1: Getting red/blue dots for anchor piece (P{anchor_piece_id})...")

            # Get the pre-calculated blue/red dots from chain matching
            if best_chain.piece1_id == anchor_piece_id:
                B1_original = best_chain.blue_dot_p1.copy()
                R1_original = best_chain.red_dot_p1.copy()
            else:
                B1_original = best_chain.blue_dot_p2.copy()
                R1_original = best_chain.red_dot_p2.copy()

            # Apply transformations to get placed position
            # CRITICAL: Use the anchor's FINAL rotation (includes additional rotation from Step 4)
            anchor_centroid = piece_centroids[anchor_piece_id]
            anchor_final_rotation = anchor_placed.rotation  # Use actual placed rotation!

            # 1. Rotate around centroid with FINAL rotation
            theta = np.radians(anchor_final_rotation)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            B1_translated = B1_original - anchor_centroid
            B1_rotated = rotation_matrix @ B1_translated
            B1_after_rotation = B1_rotated + anchor_centroid

            R1_translated = R1_original - anchor_centroid
            R1_rotated = rotation_matrix @ R1_translated
            R1_after_rotation = R1_rotated + anchor_centroid

            # 2. Apply the same translation as the placed piece
            translation = anchor_placed.position - anchor_centroid
            B1_placed = B1_after_rotation + translation
            R1_placed = R1_after_rotation + translation

            print(f"    Using anchor's final rotation: {anchor_final_rotation:.1f} deg (not just normalization)")

            print(f"    Blue dot (B1) at: ({B1_placed[0]:.1f}, {B1_placed[1]:.1f})")
            print(f"    Red dot (R1) at: ({R1_placed[0]:.1f}, {R1_placed[1]:.1f})")

            # =================================================================
            # STEP 2: Get next piece chain red/blue dots (from pre-calculated values)
            # =================================================================
            print(f"\n  Step 2: Getting red/blue dots for next piece (P{next_piece_id})...")

            # Get the pre-calculated blue/red dots from chain matching
            if best_chain.piece1_id == next_piece_id:
                B2_original = best_chain.blue_dot_p1.copy()
                R2_original = best_chain.red_dot_p1.copy()
            else:
                B2_original = best_chain.blue_dot_p2.copy()
                R2_original = best_chain.red_dot_p2.copy()

            # Apply normalization rotation to next piece dots
            next_centroid = piece_centroids[next_piece_id]
            next_base_rotation = piece_orientations[next_piece_id]

            theta = np.radians(next_base_rotation)
            rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])

            B2_translated = B2_original - next_centroid
            B2_rotated = rotation_matrix @ B2_translated
            B2_normalized = B2_rotated + next_centroid

            R2_translated = R2_original - next_centroid
            R2_rotated = rotation_matrix @ R2_translated
            R2_normalized = R2_rotated + next_centroid

            print(f"    Blue dot (B2) at: ({B2_normalized[0]:.1f}, {B2_normalized[1]:.1f})")
            print(f"    Red dot (R2) at: ({R2_normalized[0]:.1f}, {R2_normalized[1]:.1f})")

            # =================================================================
            # STEP 3: Align chains using red/blue dots
            # =================================================================
            print(f"\n  Step 3: Aligning chains (B2->B1, R2->R1)...")

            # Get the normalized contour of the next piece (already rotated for axis alignment)
            next_normalized = normalized_pieces[next_piece_id]
            next_contour_normalized = next_normalized['normalized_contour']

            # Step 3.1: Translate so B2 aligns with B1
            translation = B1_placed - B2_normalized
            next_contour_translated = next_contour_normalized + translation
            B2_translated = B2_normalized + translation
            R2_translated = R2_normalized + translation

            print(f"    Translation: ({translation[0]:.1f}, {translation[1]:.1f})")
            print(f"    After translation - B2: ({B2_translated[0]:.1f}, {B2_translated[1]:.1f})")
            print(f"    After translation - R2: ({R2_translated[0]:.1f}, {R2_translated[1]:.1f})")

            # Step 3.2: Rotate around B1 to align R2 with R1
            from ..edge_solver_v2.geometry_utils import GeometryUtils

            # Calculate rotation angle needed
            vec_current = R2_translated - B1_placed
            vec_desired = R1_placed - B1_placed

            if np.linalg.norm(vec_current) > 0 and np.linalg.norm(vec_desired) > 0:
                angle_current = np.arctan2(vec_current[1], vec_current[0])
                angle_desired = np.arctan2(vec_desired[1], vec_desired[0])
                rotation_angle = angle_desired - angle_current
            else:
                rotation_angle = 0

            print(f"    Rotation angle: {np.degrees(rotation_angle):.1f}°")

            # Apply rotation to all points around B1
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a],
                [sin_a, cos_a]
            ])

            # Rotate contour points
            translated_to_origin = next_contour_translated - B1_placed
            rotated = translated_to_origin @ rotation_matrix.T
            next_contour_final = rotated + B1_placed

            # Rotate R2 to verify alignment
            R2_translated_to_origin = R2_translated - B1_placed
            R2_rotated = rotation_matrix @ R2_translated_to_origin
            R2_final = R2_rotated + B1_placed

            print(f"    After rotation - B2 (should match B1): ({B1_placed[0]:.1f}, {B1_placed[1]:.1f})")
            print(f"    After rotation - R2 (should match R1): ({R2_final[0]:.1f}, {R2_final[1]:.1f})")
            print(f"    Target R1: ({R1_placed[0]:.1f}, {R1_placed[1]:.1f})")

            # Calculate final centroid and total rotation
            next_centroid_translated = next_centroid + translation
            next_centroid_to_origin = next_centroid_translated - B1_placed
            next_centroid_rotated = rotation_matrix @ next_centroid_to_origin
            final_next_centroid = next_centroid_rotated + B1_placed

            # Total rotation is base rotation + alignment rotation
            final_rotation = next_base_rotation + np.degrees(rotation_angle)

            print(f"    Final centroid: ({final_next_centroid[0]:.1f}, {final_next_centroid[1]:.1f})")
            print(f"    Final rotation: {final_rotation:.1f}°")

            # Store the placed next piece with aligned position
            placed_pieces[next_piece_id] = PlacedPiece(
                piece_id=next_piece_id,
                position=final_next_centroid,
                rotation=final_rotation,
                contour_points=next_contour_final
            )

            # Store step 5 data with red/blue dots (both before and after alignment)
            step5_data = {
                'pieces': all_pieces,
                'normalized_pieces': normalized_pieces,
                'anchor_piece_id': anchor_piece_id,
                'placed_pieces': placed_pieces.copy(),  # Make a copy
                'next_piece_id': next_piece_id,
                'best_chain': best_chain,
                'original_image': original_image,
                # Red/blue dots for anchor (already placed)
                'B1_placed': B1_placed,
                'R1_placed': R1_placed,
                # Red/blue dots for next piece (after alignment)
                'B2_final': B1_placed,  # B2 should now be at B1
                'R2_final': R2_final,   # R2 should now be at R1
            }
            assembly_steps.append(AssemblyStep(
                step_number=5,
                step_name="Place Second Piece",
                description="Place piece with highest connection score to anchor",
                visualization_data=step5_data
            ))

            print("\n[OK] Step 5 complete - second piece placed")
        else:
            print("\n[WARNING] No connections found for anchor piece, skipping Step 5")

        # STEP 6: Place third piece (second connection to anchor)
        print("\n" + "-"*70)
        print("Step 6: Placing third piece (second connection to anchor)...")
        print("-"*70)

        if len(placed_pieces) < len(all_pieces):
            print(f"\n{len(placed_pieces)} pieces placed, {len(all_pieces) - len(placed_pieces)} remaining")

            # Find the second-best connection to the anchor (best one was used in Step 5)
            # Get all anchor connections
            anchor_connections = piece_connections.get(anchor_piece_id, [])

            # Filter out the connection that was already used in Step 5
            available_anchor_connections = []
            for connected_piece_id, chain, side in anchor_connections:
                if connected_piece_id not in placed_pieces:
                    available_anchor_connections.append((connected_piece_id, chain, side))

            best_next_piece_id = None
            best_connection_to_placed = None
            best_placed_piece_id = None

            if available_anchor_connections:
                # Get the best available connection to the anchor
                best_connection = max(available_anchor_connections, key=lambda conn: conn[1].shape_score)
                best_next_piece_id, best_connection_to_placed, side = best_connection
                best_placed_piece_id = anchor_piece_id

            if best_next_piece_id is not None:
                print(f"\n  Placing piece P{best_next_piece_id}")
                print(f"  Connected to P{best_placed_piece_id} (already placed) via chain:")
                print(f"    Chain length: {best_connection_to_placed.chain_length} segments")
                print(f"    Shape score: {best_connection_to_placed.shape_score:.1f}%")
                print(f"    Segments: P{best_connection_to_placed.piece1_id}{best_connection_to_placed.segment_ids_p1} <-> P{best_connection_to_placed.piece2_id}{best_connection_to_placed.segment_ids_p2}")

                # Get the already-placed piece
                placed_piece_obj = placed_pieces[best_placed_piece_id]

                # Determine which piece is placed and which is next in the chain
                if best_connection_to_placed.piece1_id == best_placed_piece_id:
                    placed_seg_ids = best_connection_to_placed.segment_ids_p1
                    next_seg_ids = best_connection_to_placed.segment_ids_p2
                else:
                    placed_seg_ids = best_connection_to_placed.segment_ids_p2
                    next_seg_ids = best_connection_to_placed.segment_ids_p1

                # Get the pre-calculated blue/red dots from chain matching
                if best_connection_to_placed.piece1_id == best_placed_piece_id:
                    B1_original = best_connection_to_placed.blue_dot_p1.copy()
                    R1_original = best_connection_to_placed.red_dot_p1.copy()
                else:
                    B1_original = best_connection_to_placed.blue_dot_p2.copy()
                    R1_original = best_connection_to_placed.red_dot_p2.copy()

                # Transform B1/R1 to placed position
                placed_centroid_original = piece_centroids[best_placed_piece_id]
                # CRITICAL: Use the placed piece's FINAL rotation (includes additional rotation from Step 4 if it's the anchor)
                placed_final_rotation = placed_piece_obj.rotation  # Use actual placed rotation!

                # Apply rotation
                theta = np.radians(placed_final_rotation)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])
                print(f"    Using placed piece's final rotation: {placed_final_rotation:.1f} deg")

                B1_translated = B1_original - placed_centroid_original
                B1_rotated = rotation_matrix @ B1_translated
                B1_after_rotation = B1_rotated + placed_centroid_original

                R1_translated = R1_original - placed_centroid_original
                R1_rotated = rotation_matrix @ R1_translated
                R1_after_rotation = R1_rotated + placed_centroid_original

                # Apply translation
                translation = placed_piece_obj.position - placed_centroid_original
                B1_placed = B1_after_rotation + translation
                R1_placed = R1_after_rotation + translation

                # Get the pre-calculated blue/red dots for next piece
                if best_connection_to_placed.piece1_id == best_next_piece_id:
                    B2_original = best_connection_to_placed.blue_dot_p1.copy()
                    R2_original = best_connection_to_placed.red_dot_p1.copy()
                else:
                    B2_original = best_connection_to_placed.blue_dot_p2.copy()
                    R2_original = best_connection_to_placed.red_dot_p2.copy()

                # Apply normalization rotation to next piece dots
                next_centroid = piece_centroids[best_next_piece_id]
                next_base_rotation = piece_orientations[best_next_piece_id]

                theta = np.radians(next_base_rotation)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

                B2_translated = B2_original - next_centroid
                B2_rotated = rotation_matrix @ B2_translated
                B2_normalized = B2_rotated + next_centroid

                R2_translated = R2_original - next_centroid
                R2_rotated = rotation_matrix @ R2_translated
                R2_normalized = R2_rotated + next_centroid

                # Align chains (B2->B1, R2->R1)
                next_normalized = normalized_pieces[best_next_piece_id]
                next_contour_normalized = next_normalized['normalized_contour']

                # Step 1: Translate so B2 aligns with B1
                translation_align = B1_placed - B2_normalized
                next_contour_translated = next_contour_normalized + translation_align
                B2_translated = B2_normalized + translation_align
                R2_translated = R2_normalized + translation_align

                # Step 2: Rotate around B1 to align R2 with R1
                vec_current = R2_translated - B1_placed
                vec_desired = R1_placed - B1_placed

                if np.linalg.norm(vec_current) > 0 and np.linalg.norm(vec_desired) > 0:
                    angle_current = np.arctan2(vec_current[1], vec_current[0])
                    angle_desired = np.arctan2(vec_desired[1], vec_desired[0])
                    rotation_angle = angle_desired - angle_current
                else:
                    rotation_angle = 0

                # Apply rotation to all points around B1
                cos_a = np.cos(rotation_angle)
                sin_a = np.sin(rotation_angle)
                rotation_matrix_align = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])

                # Rotate contour points
                translated_to_origin = next_contour_translated - B1_placed
                rotated = translated_to_origin @ rotation_matrix_align.T
                next_contour_final = rotated + B1_placed

                # Calculate final centroid and total rotation
                next_centroid_translated = next_centroid + translation_align
                next_centroid_to_origin = next_centroid_translated - B1_placed
                next_centroid_rotated = rotation_matrix_align @ next_centroid_to_origin
                final_next_centroid = next_centroid_rotated + B1_placed

                # Total rotation is base rotation + alignment rotation
                final_rotation = next_base_rotation + np.degrees(rotation_angle)

                print(f"    Aligned P{best_next_piece_id} to P{best_placed_piece_id}")
                print(f"    Final centroid: ({final_next_centroid[0]:.1f}, {final_next_centroid[1]:.1f})")
                print(f"    Final rotation: {final_rotation:.1f}°")

                # Store the placed piece
                placed_pieces[best_next_piece_id] = PlacedPiece(
                    piece_id=best_next_piece_id,
                    position=final_next_centroid,
                    rotation=final_rotation,
                    contour_points=next_contour_final
                )

                # Calculate R2_final for visualization
                R2_translated_to_origin = R2_translated - B1_placed
                R2_rotated = rotation_matrix_align @ R2_translated_to_origin
                R2_final = R2_rotated + B1_placed

                # Store step 6 data
                step6_data = {
                    'pieces': all_pieces,
                    'normalized_pieces': normalized_pieces,
                    'placed_piece_id': best_placed_piece_id,
                    'placed_pieces': placed_pieces.copy(),
                    'next_piece_id': best_next_piece_id,
                    'best_chain': best_connection_to_placed,
                    'original_image': original_image,
                    'B1_placed': B1_placed,
                    'R1_placed': R1_placed,
                    'B2_final': B1_placed,
                    'R2_final': R2_final,
                }
                assembly_steps.append(AssemblyStep(
                    step_number=6,
                    step_name="Place Third Piece",
                    description="Place third piece using chain alignment",
                    visualization_data=step6_data
                ))

                print("\n[OK] Step 6 complete - third piece placed")
            else:
                print("\n[WARNING] No connection found for third piece")
        else:
            print("\n[WARNING] All pieces already placed, skipping Step 6")

        # STEP 7: Place fourth piece (final piece)
        print("\n" + "-"*70)
        print("Step 7: Placing fourth piece (final piece)...")
        print("-"*70)

        if len(placed_pieces) < len(all_pieces):
            print(f"\n{len(placed_pieces)} pieces placed, {len(all_pieces) - len(placed_pieces)} remaining")

            # TREE APPROACH: The 4th piece must connect to one of the anchor's children (Pieces 2 or 3)
            # Not to the anchor itself, as the anchor's 2 connection slots are already used

            # Get the list of child pieces (pieces connected to anchor, excluding the anchor)
            child_piece_ids = [pid for pid in placed_pieces.keys() if pid != anchor_piece_id]

            print(f"  Considering connections to child pieces: {child_piece_ids}")

            # Find the best unplaced piece to place next (only consider connections to children)
            best_next_piece_id = None
            best_connection_to_placed = None
            best_placed_piece_id = None
            best_score = -1

            # Check all unplaced pieces
            for piece_id in range(len(all_pieces)):
                if piece_id in placed_pieces:
                    continue  # Already placed

                # Check connections to child pieces ONLY (not the anchor)
                connections = piece_connections.get(piece_id, [])
                for connected_piece_id, chain, side in connections:
                    if connected_piece_id in child_piece_ids:
                        # This unplaced piece connects to a child piece
                        if chain.shape_score > best_score:
                            best_score = chain.shape_score
                            best_next_piece_id = piece_id
                            best_connection_to_placed = chain
                            best_placed_piece_id = connected_piece_id

            if best_next_piece_id is not None:
                print(f"\n  Placing piece P{best_next_piece_id}")
                print(f"  Connected to P{best_placed_piece_id} (child piece) via chain:")
                print(f"    Chain length: {best_connection_to_placed.chain_length} segments")
                print(f"    Shape score: {best_connection_to_placed.shape_score:.1f}%")
                print(f"    Segments: P{best_connection_to_placed.piece1_id}{best_connection_to_placed.segment_ids_p1} <-> P{best_connection_to_placed.piece2_id}{best_connection_to_placed.segment_ids_p2}")

                # Get the already-placed piece
                placed_piece_obj = placed_pieces[best_placed_piece_id]

                # Determine which piece is placed and which is next in the chain
                if best_connection_to_placed.piece1_id == best_placed_piece_id:
                    placed_seg_ids = best_connection_to_placed.segment_ids_p1
                    next_seg_ids = best_connection_to_placed.segment_ids_p2
                else:
                    placed_seg_ids = best_connection_to_placed.segment_ids_p2
                    next_seg_ids = best_connection_to_placed.segment_ids_p1

                # Get the pre-calculated blue/red dots from chain matching
                if best_connection_to_placed.piece1_id == best_placed_piece_id:
                    B1_original = best_connection_to_placed.blue_dot_p1.copy()
                    R1_original = best_connection_to_placed.red_dot_p1.copy()
                else:
                    B1_original = best_connection_to_placed.blue_dot_p2.copy()
                    R1_original = best_connection_to_placed.red_dot_p2.copy()

                # Transform B1/R1 to the ACTUAL placed position
                # CRITICAL FIX: Use placed_piece_obj.rotation (final rotation after placement)
                # NOT piece_orientations[best_placed_piece_id] (only normalization rotation)
                placed_centroid_original = piece_centroids[best_placed_piece_id]
                placed_final_rotation = placed_piece_obj.rotation  # Actual final rotation!

                # Apply the FINAL rotation
                theta = np.radians(placed_final_rotation)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

                B1_translated = B1_original - placed_centroid_original
                B1_rotated = rotation_matrix @ B1_translated
                B1_after_rotation = B1_rotated + placed_centroid_original

                R1_translated = R1_original - placed_centroid_original
                R1_rotated = rotation_matrix @ R1_translated
                R1_after_rotation = R1_rotated + placed_centroid_original

                # Apply translation to final placed position
                translation = placed_piece_obj.position - placed_centroid_original
                B1_placed = B1_after_rotation + translation
                R1_placed = R1_after_rotation + translation

                # Get the pre-calculated blue/red dots for next piece
                if best_connection_to_placed.piece1_id == best_next_piece_id:
                    B2_original = best_connection_to_placed.blue_dot_p1.copy()
                    R2_original = best_connection_to_placed.red_dot_p1.copy()
                else:
                    B2_original = best_connection_to_placed.blue_dot_p2.copy()
                    R2_original = best_connection_to_placed.red_dot_p2.copy()

                # Apply normalization rotation to next piece dots
                next_centroid = piece_centroids[best_next_piece_id]
                next_base_rotation = piece_orientations[best_next_piece_id]

                theta = np.radians(next_base_rotation)
                rotation_matrix = np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ])

                B2_translated = B2_original - next_centroid
                B2_rotated = rotation_matrix @ B2_translated
                B2_normalized = B2_rotated + next_centroid

                R2_translated = R2_original - next_centroid
                R2_rotated = rotation_matrix @ R2_translated
                R2_normalized = R2_rotated + next_centroid

                # Align chains (B2->B1, R2->R1)
                next_normalized = normalized_pieces[best_next_piece_id]
                next_contour_normalized = next_normalized['normalized_contour']

                # Step 1: Translate so B2 aligns with B1
                translation_align = B1_placed - B2_normalized
                next_contour_translated = next_contour_normalized + translation_align
                B2_translated = B2_normalized + translation_align
                R2_translated = R2_normalized + translation_align

                # Step 2: Rotate around B1 to align R2 with R1
                vec_current = R2_translated - B1_placed
                vec_desired = R1_placed - B1_placed

                if np.linalg.norm(vec_current) > 0 and np.linalg.norm(vec_desired) > 0:
                    angle_current = np.arctan2(vec_current[1], vec_current[0])
                    angle_desired = np.arctan2(vec_desired[1], vec_desired[0])
                    rotation_angle = angle_desired - angle_current
                else:
                    rotation_angle = 0

                # Apply rotation to all points around B1
                cos_a = np.cos(rotation_angle)
                sin_a = np.sin(rotation_angle)
                rotation_matrix_align = np.array([
                    [cos_a, -sin_a],
                    [sin_a, cos_a]
                ])

                # Rotate contour points
                translated_to_origin = next_contour_translated - B1_placed
                rotated = translated_to_origin @ rotation_matrix_align.T
                next_contour_final = rotated + B1_placed

                # Calculate final centroid and total rotation
                next_centroid_translated = next_centroid + translation_align
                next_centroid_to_origin = next_centroid_translated - B1_placed
                next_centroid_rotated = rotation_matrix_align @ next_centroid_to_origin
                final_next_centroid = next_centroid_rotated + B1_placed

                # Total rotation is base rotation + alignment rotation
                final_rotation = next_base_rotation + np.degrees(rotation_angle)

                print(f"    Aligned P{best_next_piece_id} to P{best_placed_piece_id}")
                print(f"    Final centroid: ({final_next_centroid[0]:.1f}, {final_next_centroid[1]:.1f})")
                print(f"    Final rotation: {final_rotation:.1f}°")

                # Store the placed piece
                placed_pieces[best_next_piece_id] = PlacedPiece(
                    piece_id=best_next_piece_id,
                    position=final_next_centroid,
                    rotation=final_rotation,
                    contour_points=next_contour_final
                )

                # Calculate R2_final for visualization
                R2_translated_to_origin = R2_translated - B1_placed
                R2_rotated = rotation_matrix_align @ R2_translated_to_origin
                R2_final = R2_rotated + B1_placed

                # Store step 7 data
                step7_data = {
                    'pieces': all_pieces,
                    'normalized_pieces': normalized_pieces,
                    'placed_piece_id': best_placed_piece_id,
                    'placed_pieces': placed_pieces.copy(),
                    'next_piece_id': best_next_piece_id,
                    'best_chain': best_connection_to_placed,
                    'original_image': original_image,
                    'B1_placed': B1_placed,
                    'R1_placed': R1_placed,
                    'B2_final': B1_placed,
                    'R2_final': R2_final,
                }
                assembly_steps.append(AssemblyStep(
                    step_number=7,
                    step_name="Place Fourth Piece",
                    description="Place final piece using chain alignment",
                    visualization_data=step7_data
                ))

                print("\n[OK] Step 7 complete - fourth piece placed")
            else:
                print("\n[WARNING] No connection found for fourth piece")
        else:
            print("\n[WARNING] All pieces already placed, skipping Step 7")

        # STEP 8: Final assembly summary
        print("\n" + "-"*70)
        print("Step 8: Final assembly summary...")
        print("-"*70)

        print(f"\n{len(placed_pieces)}/{len(all_pieces)} pieces successfully placed")

                # Add final assembly step showing all pieces
        step6_data = {
            'pieces': all_pieces,
            'placed_pieces': placed_pieces.copy(),
            'original_image': original_image
        }
        assembly_steps.append(AssemblyStep(
            step_number=8,
            step_name="Final Assembly",
            description="All pieces placed together",
            visualization_data=step6_data
        ))

        # Return results
        return {
            'assembly_steps': assembly_steps,
            'piece_centroids': piece_centroids,
            'piece_orientations': piece_orientations,
            'normalized_pieces': normalized_pieces,
            'piece_connections': piece_connections,
            'piece_scores': piece_scores,
            'anchor_piece_id': anchor_piece_id,
            'placed_pieces': placed_pieces,
        }
