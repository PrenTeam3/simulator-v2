from pathlib import Path
import sys
import uuid
from datetime import datetime
import cv2
from puzzle_finder import detect_a4_area, warp_a4_region
from puzzle_analyzer.piece_detector import detect_pieces, draw_pieces
from puzzle_analyzer.corner_detector import detect_corners, draw_corners
from puzzle_analyzer.piece_classifier import classify_piece, draw_classification, draw_debug, draw_outside_segments_debug
from puzzle_solverv2.frame import build_frame
from puzzle_solverv2.border_info import extract_border_info, log_border_info
from puzzle_solverv2.variants import generate_variants, log_variants
from puzzle_solverv2.similarity import analyze_similarity, log_similarity
from puzzle_solverv2.constraints import log_constraints
from puzzle_solverv2.tree_search import visualize_start_placements, visualize_second_placements


# ─────────────────────────────────────────────
#  Tee — write stdout to console AND log file
# ─────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
    def flush(self):
        for s in self._streams:
            s.flush()


# ─────────────────────────────────────────────
#  Logging helper
# ─────────────────────────────────────────────

def log_step(number, title):
    print(f"\n{'='*60}")
    print(f"  STEP {number}: {title}")
    print(f"{'='*60}")

def log(msg):
    print(f"  {msg}")

def log_ok(msg):
    print(f"  [OK] {msg}")

def log_err(msg):
    print(f"  [ERROR] {msg}")

def log_out(path):
    print(f"  [OUT] {path}")


# ─────────────────────────────────────────────
#  Run setup
# ─────────────────────────────────────────────

image_path = "../images/new/test7.jpg"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
output_dir = Path("output") / run_id
output_dir.mkdir(parents=True, exist_ok=True)

_log_file = open(output_dir / "run.log", "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log_file)

print(f"\n{'#'*60}")
print(f"  Puzzle Solver v2")
print(f"  Run ID : {run_id}")
print(f"  Output : {output_dir}/")
print(f"  Image  : {image_path}")
print(f"{'#'*60}")


# ─────────────────────────────────────────────
#  [Analyzer] Find and rectify the A4 area
# ─────────────────────────────────────────────

log_step("A1", "Detect and rectify A4 area")

image = cv2.imread(image_path)
if image is None:
    log_err(f"Could not load image: {image_path}")
    exit(1)
log_ok(f"Image loaded: {image.shape[1]}x{image.shape[0]}px")

detection = detect_a4_area(image)
if detection is None:
    log_err("No A4 area detected — aborting.")
    exit(2)
log_ok("A4 area detected")

a4_image = warp_a4_region(image, detection)
cv2.imwrite(str(output_dir / "a4_rectified.png"), a4_image)
log_ok(f"A4 rectified: {a4_image.shape[1]}x{a4_image.shape[0]}px")
log_out("a4_rectified.png")


# ─────────────────────────────────────────────
#  [Analyzer] Detect puzzle pieces
# ─────────────────────────────────────────────

log_step("A2", "Detect puzzle pieces")

contours, thresh = detect_pieces(a4_image)
cv2.imwrite(str(output_dir / "threshold.png"), thresh)
log_out("threshold.png")

VALID_PIECE_COUNTS = {4, 6}
if len(contours) not in VALID_PIECE_COUNTS:
    log_err(f"Expected 4 or 6 pieces, found {len(contours)} — aborting.")
    exit(3)
log_ok(f"{len(contours)} pieces detected")

annotated = draw_pieces(a4_image, contours)
cv2.imwrite(str(output_dir / "pieces_detected.png"), annotated)
log_out("pieces_detected.png")


# ─────────────────────────────────────────────
#  [Analyzer] Detect corners per piece
# ─────────────────────────────────────────────

log_step("A3", "Detect corners per piece")

corners_list = [detect_corners(c) for c in contours]

for idx, info in enumerate(corners_list):
    n_straight = sum(1 for s in info['all_segments'] if s['is_straight'])
    log(f"Piece {idx}: {len(info['outer_corners'])} outer corners, "
        f"{len(info['inner_corners'])} inner corners, "
        f"{n_straight} straight segments")

corners_image = draw_corners(a4_image, contours, corners_list)
cv2.imwrite(str(output_dir / "corners_detected.png"), corners_image)
log_out("corners_detected.png")


# ─────────────────────────────────────────────
#  [Analyzer] Classify pieces
# ─────────────────────────────────────────────

log_step("A4", "Classify pieces (corner / edge / inner)")

classifications = [classify_piece(info, debug=True, piece_idx=idx) for idx, info in enumerate(corners_list)]

for idx, cls in enumerate(classifications):
    fc = len(cls['frame_corners'])
    log(f"Piece {idx}: type={cls['type']}  "
        f"straight={cls['num_straight']}  frame_corners={fc}")

classified_image = draw_classification(a4_image, contours, corners_list, classifications)
cv2.imwrite(str(output_dir / "classified.png"), classified_image)
log_out("classified.png")

debug_image = draw_debug(a4_image, contours, corners_list, classifications)
cv2.imwrite(str(output_dir / "debug.png"), debug_image)
log_out("debug.png")

outside_image = draw_outside_segments_debug(a4_image, contours, corners_list, classifications)
cv2.imwrite(str(output_dir / "outside_segments.png"), outside_image)
log_out("outside_segments.png")


# ─────────────────────────────────────────────
#  [Solver v2] Step 1 — Define puzzle frame
# ─────────────────────────────────────────────

log_step(1, "Define puzzle frame")

frame = build_frame(a4_image.shape[1])
log_ok(f"Frame: {frame.width_mm}mm wide x {frame.height_mm}mm tall")
log_ok(f"Scale: {frame.px_per_mm:.4f} px/mm  (from {a4_image.shape[1]}px / 297mm)")


# ─────────────────────────────────────────────
#  [Solver v2] Step 2 — Extract border info
# ─────────────────────────────────────────────

log_step(2, "Extract border info per piece")

pieces_border = extract_border_info(corners_list, classifications, frame)
log_border_info(pieces_border)
log_ok(f"{len(pieces_border)} pieces processed")


# ─────────────────────────────────────────────
#  [Solver v2] Step 3 — Generate variants per piece
# ─────────────────────────────────────────────

log_step(3, "Generate variants per piece")

pieces_variants = generate_variants(pieces_border)
log_variants(pieces_variants)
log_ok(f"{sum(len(pv.variants) for pv in pieces_variants)} total variants across {len(pieces_variants)} pieces")


# ─────────────────────────────────────────────
#  [Solver v2] Step 4 — Similarity analysis
# ─────────────────────────────────────────────

log_step(4, "Pairwise similarity analysis")

similarity_results = analyze_similarity(contours)
log_similarity(similarity_results)
duplicates = [r for r in similarity_results if r.likely_duplicate]
log_ok(f"{len(similarity_results)} pairs compared, {len(duplicates)} likely duplicate(s) found")


# ─────────────────────────────────────────────
#  [Solver v2] Step 5 — Define constraints
# ─────────────────────────────────────────────

log_step(5, "Constraint configuration")

log_constraints(frame)
log_ok("Constraints defined — will be applied in Step 6 (tree search)")


# ─────────────────────────────────────────────
#  [Solver v2] Step 6 debug — Visual placement check
# ─────────────────────────────────────────────

log_step("6-debug", "Visual placement check (TL + BL per piece)")

visualize_start_placements(pieces_border, pieces_variants, corners_list, frame, output_dir)
log_ok("Step 1 placement images saved (dbg_place_P*)")

visualize_second_placements(pieces_border, pieces_variants, corners_list, frame, output_dir)
log_ok("Step 2 placement images saved (dbg_step2_P*)")



# ─────────────────────────────────────────────
#  Done
# ─────────────────────────────────────────────

print(f"\n{'#'*60}")
print(f"  Run complete — outputs in: {output_dir}/")
print(f"{'#'*60}\n")

sys.stdout = sys.__stdout__
_log_file.close()
