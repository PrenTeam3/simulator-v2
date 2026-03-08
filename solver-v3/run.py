from pathlib import Path
import cv2
from puzzle_finder import detect_a4_area, warp_a4_region
from puzzle_analyzer.piece_detector import detect_pieces, draw_pieces
from puzzle_analyzer.corner_detector import detect_corners, draw_corners
from puzzle_analyzer.piece_classifier import classify_piece, draw_classification, draw_debug, draw_outside_segments_debug

image_path = "../images/new/test7.jpg"
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Find and rectify the A4 area
image = cv2.imread(image_path)
if image is None:
    print(f"Could not load image: {image_path}")
    exit(1)

detection = detect_a4_area(image)
if detection is None:
    print("No A4 area detected.")
    exit(2)

a4_image = warp_a4_region(image, detection)
cv2.imwrite(str(output_dir / "a4_rectified.png"), a4_image)
print(f"A4 rectified: {a4_image.shape[1]}x{a4_image.shape[0]}px")

# Step 2: Detect puzzle pieces on the A4 image
contours, thresh = detect_pieces(a4_image)

cv2.imwrite(str(output_dir / "threshold.png"), thresh)

VALID_PIECE_COUNTS = {4, 6}
if len(contours) not in VALID_PIECE_COUNTS:
    print(f"Error: Expected 4 or 6 pieces, but found {len(contours)}. Aborting.")
    exit(3)

annotated = draw_pieces(a4_image, contours)
cv2.imwrite(str(output_dir / "pieces_detected.png"), annotated)

# Step 3 & 4: Detect and visualize corners per piece
corners_list = [detect_corners(c) for c in contours]

for idx, info in enumerate(corners_list):
    n_straight = sum(1 for s in info['all_segments'] if s['is_straight'])
    print(f"  Piece {idx}: {len(info['outer_corners'])} outer, "
          f"{len(info['inner_corners'])} inner, "
          f"{n_straight} straight edges")

corners_image = draw_corners(a4_image, contours, corners_list)
cv2.imwrite(str(output_dir / "corners_detected.png"), corners_image)

# Step 5: Classify pieces
classifications = [classify_piece(info, debug=True, piece_idx=idx) for idx, info in enumerate(corners_list)]

for idx, cls in enumerate(classifications):
    fc = len(cls['frame_corners'])
    print(f"  Piece {idx}: {cls['type']}  "
          f"({cls['num_straight']} straight, {fc} frame corner(s))")

classified_image = draw_classification(a4_image, contours, corners_list, classifications)
cv2.imwrite(str(output_dir / "classified.png"), classified_image)

debug_image = draw_debug(a4_image, contours, corners_list, classifications)
cv2.imwrite(str(output_dir / "debug.png"), debug_image)

outside_image = draw_outside_segments_debug(a4_image, contours, corners_list, classifications)
cv2.imwrite(str(output_dir / "outside_segments.png"), outside_image)

print(f"Outputs saved to: {output_dir}/")
