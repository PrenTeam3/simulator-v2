import os
import json
from pathlib import Path
from datetime import datetime
import cv2


def save_analysis_results(output_image, puzzle_pieces, image_path, debug_visualization=False):
    """
    Save puzzle analysis results to a temporary folder in the repository.

    Args:
        output_image: The analyzed image with visualizations (numpy array)
        puzzle_pieces: List of PuzzlePiece objects
        image_path: Path to the original image file
        debug_visualization: Whether debug visualization was enabled

    Returns:
        str: Path to the temporary folder where results were saved
    """
    # Create a temporary directory inside the repo with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    repo_root = Path(__file__).parent.parent  # Navigate to repo root
    temp_dir = repo_root / "temp" / f"puzzle_analysis_{timestamp}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = str(temp_dir)

    print(f"\nSaving results to: {temp_dir}")

    # 1. Save the visualization image
    output_image_path = os.path.join(temp_dir, "analysis_results.png")
    cv2.imwrite(output_image_path, output_image)
    print(f"  [OK] Saved visualization: {output_image_path}")

    # 2. Save detailed analysis data as JSON
    analysis_data = {
        "timestamp": timestamp,
        "image_path": image_path,
        "total_pieces": len(puzzle_pieces),
        "debug_visualization_enabled": debug_visualization,
        "pieces": []
    }

    total_border_edges = 0
    total_frame_corners = 0

    for i, piece in enumerate(puzzle_pieces):
        # Full contour points (raw contour recognition)
        contour_points = []
        if piece.contour is not None:
            for point in piece.contour[:, 0, :]:
                contour_points.append({"x": float(point[0]), "y": float(point[1])})

        # Approximate polygon points
        approx_poly_points = []
        if piece.approx_poly is not None:
            for point in piece.approx_poly[:, 0, :]:
                approx_poly_points.append({"x": float(point[0]), "y": float(point[1])})

        # Convex hull points
        convex_hull_points = []
        if piece.convex_hull is not None:
            for point in piece.convex_hull[:, 0, :]:
                convex_hull_points.append({"x": float(point[0]), "y": float(point[1])})

        piece_data = {
            "piece_id": i,
            "area": float(piece.area),
            "bounding_box": {
                "x": int(piece.bbox[0]),
                "y": int(piece.bbox[1]),
                "width": int(piece.bbox[2]),
                "height": int(piece.bbox[3])
            },
            "centroid": {
                "x": float(piece.centroid[0]),
                "y": float(piece.centroid[1])
            },
            "contour_recognition": {
                "contour_points_count": len(contour_points),
                "approximate_polygon_points_count": len(approx_poly_points),
                "convex_hull_points_count": len(convex_hull_points),
                "contour_points": contour_points,
                "approximate_polygon_points": approx_poly_points,
                "convex_hull_points": convex_hull_points
            },
            "corners": {
                "outer_corners_count": len(piece.outer_corners),
                "inner_corners_count": len(piece.inner_corners),
                "curved_points_count": len(piece.curved_points),
                "frame_corners_count": len(piece.frame_corners)
            },
            "features": {
                "straight_segments_count": len(piece.straight_segments),
                "border_edges_count": len(piece.border_edges)
            },
            "outer_corners": [{"x": float(c[0]), "y": float(c[1])} for c in piece.outer_corners],
            "inner_corners": [{"x": float(c[0]), "y": float(c[1])} for c in piece.inner_corners],
            "curved_points": [{"x": float(c[0]), "y": float(c[1])} for c in piece.curved_points],
            "frame_corners": [{"x": float(c[0]), "y": float(c[1])} for c in piece.frame_corners]
        }

        # Add straight segments data
        segments_data = []
        for seg in piece.straight_segments:
            segments_data.append({
                "p1": {"x": float(seg.p1[0]), "y": float(seg.p1[1])},
                "p2": {"x": float(seg.p2[0]), "y": float(seg.p2[1])},
                "length": float(seg.length),
                "is_border_edge": seg.is_border_edge
            })
        piece_data["straight_segments"] = segments_data

        # Add border edges data
        border_edges_data = []
        for seg in piece.border_edges:
            border_edges_data.append({
                "p1": {"x": float(seg.p1[0]), "y": float(seg.p1[1])},
                "p2": {"x": float(seg.p2[0]), "y": float(seg.p2[1])},
                "length": float(seg.length)
            })
        piece_data["border_edges"] = border_edges_data

        analysis_data["pieces"].append(piece_data)

        total_border_edges += len(piece.border_edges)
        total_frame_corners += len(piece.frame_corners)

    analysis_data["summary"] = {
        "total_border_edges": total_border_edges,
        "total_frame_corners": total_frame_corners
    }

    json_path = os.path.join(temp_dir, "analysis_data.json")
    with open(json_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    print(f"  [OK] Saved analysis data: {json_path}")

    # 3. Save a summary report as text
    report_path = os.path.join(temp_dir, "analysis_report.txt")
    with open(report_path, 'w') as f:
        f.write("PUZZLE ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Debug Visualization: {'ENABLED' if debug_visualization else 'DISABLED'}\n\n")

        f.write("FINAL SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total puzzle pieces analyzed: {len(puzzle_pieces)}\n")
        f.write(f"Total border edges found: {total_border_edges}\n")
        f.write(f"Total frame corners found: {total_frame_corners}\n\n")

        f.write("PIECES DETAILED ANALYSIS\n")
        f.write("-" * 70 + "\n\n")

        for i, piece in enumerate(puzzle_pieces):
            f.write(f"Piece {i}\n")
            f.write(f"  Area: {piece.area} pixels²\n")
            f.write(f"  Bounding Box: x={piece.bbox[0]}, y={piece.bbox[1]}, width={piece.bbox[2]}, height={piece.bbox[3]}\n")
            f.write(f"  Centroid (Grip Position): ({piece.centroid[0]:.1f}, {piece.centroid[1]:.1f})\n")
            f.write(f"  Contour Recognition:\n")
            f.write(f"    - Raw contour points: {len(piece.contour[:, 0, :])}\n")
            f.write(f"    - Approximate polygon points: {len(piece.approx_poly[:, 0, :])}\n")
            f.write(f"    - Convex hull points: {len(piece.convex_hull[:, 0, :])}\n")
            f.write(f"  Corners:\n")
            f.write(f"    - Outer corners: {len(piece.outer_corners)}\n")
            f.write(f"    - Inner corners: {len(piece.inner_corners)}\n")
            f.write(f"    - Curved transitions: {len(piece.curved_points)}\n")
            f.write(f"    - Frame corners (90deg): {len(piece.frame_corners)}\n")
            f.write(f"  Features:\n")
            f.write(f"    - Straight segments: {len(piece.straight_segments)}\n")
            f.write(f"    - Border edges: {len(piece.border_edges)}\n\n")

    print(f"  [OK] Saved report: {report_path}")

    print(f"\n[OK] All results saved successfully to: {temp_dir}\n")

    return temp_dir
