# Puzzle Solver V2 - Complete Documentation

A sophisticated jigsaw puzzle solving algorithm that analyzes puzzle pieces from images and determines how they fit together. The system uses computer vision, geometric analysis, and intelligent matching algorithms to solve puzzles automatically.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Phase 1: Puzzle Analysis](#phase-1-puzzle-analysis)
4. [Phase 2: Puzzle Solving](#phase-2-puzzle-solving)
5. [Quick Start Guide](#quick-start-guide)
6. [Detailed Process Flow](#detailed-process-flow)
7. [Algorithm Details](#algorithm-details)
8. [Configuration Options](#configuration-options)
9. [Output Files](#output-files)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The Puzzle Solver V2 is a two-phase system:

1. **Analysis Phase** - Detects puzzle pieces in an image and analyzes their geometric properties
2. **Solving Phase** - Matches pieces together using sophisticated algorithms

### Key Features

- Automatic puzzle piece detection from images
- Intelligent corner and edge detection
- Frame corner identification (border pieces)
- Two solving algorithms: Matrix-based and Edge-based
- Comprehensive visualization at each step
- Adaptive strictness for robust detection
- SVG-based geometric analysis for high accuracy

---

## System Architecture

```
version-v2/
├── analyze.py                    # Entry point for analysis phase
├── solve.py                      # Entry point for solving phase
│
├── puzzle_analyzer_v2/           # Phase 1: Piece Detection & Analysis
│   ├── core.py                   # Main analyzer orchestration
│   ├── corner_detector.py        # Corner detection & classification
│   ├── svg_visualizer.py         # SVG generation from contours
│   ├── svg_smoother.py           # Contour smoothing
│   ├── svg_corner_drawer.py      # Corner marking on SVG
│   └── ...
│
├── puzzle_solver_v2/             # Phase 2: Puzzle Solving
│   ├── solver.py                 # Main solver orchestration
│   │
│   ├── preparation/              # Data loading & preparation
│   │   ├── prepare.py            # Preparation orchestration
│   │   ├── data_loader.py        # Load analysis data
│   │   ├── svg_data_loader.py    # Parse SVG files
│   │   ├── contour_segmenter.py  # Extract segments
│   │   └── visualizer.py         # Initial visualization
│   │
│   ├── matrix_solver/            # Algorithm 1: Matrix-based solving
│   │   ├── matrix_solver.py      # Matrix solver orchestration
│   │   ├── segment_matcher.py    # Compare segments
│   │   ├── diagonal_extractor.py # Find matching patterns
│   │   ├── group_validator.py    # Validate matches
│   │   └── ...visualizers        # Visualization utilities
│   │
│   ├── edge_solver/              # Algorithm 2: Edge-based solving
│   │   ├── edge_solver.py        # Edge solver orchestration
│   │   ├── segment_finder.py     # Find frame-adjacent segments
│   │   ├── frame_adjacent_matcher.py  # Match segments
│   │   ├── connection_manager.py # Determine connections
│   │   ├── rotation_calculator.py # Calculate rotations
│   │   ├── solution_builder.py   # Build final solution
│   │   └── visualizers.py        # Visualization utilities
│   │
│   └── common/                   # Shared utilities
│       ├── data_classes.py       # Data structures
│       ├── utils.py              # Helper functions
│       └── image_viewer.py       # Interactive viewer
│
└── temp/                         # Output directory
    └── analysis_YYYYMMDD_HHMMSS/ # Timestamped results
```

---

## Phase 1: Puzzle Analysis

**Purpose:** Detect and analyze puzzle pieces from an input image

### 1.1 Input

- **Image file** containing puzzle pieces (JPG, PNG, etc.)
- Pieces should be separated with clear background
- Works with both light and dark backgrounds

### 1.2 Process

#### Step 1: Piece Detection (core.py:63-104)

1. Convert image to grayscale
2. Apply Gaussian blur for noise reduction
3. Automatic background brightness detection
4. Otsu's thresholding (adaptive based on background)
5. Morphological operations to clean up noise
6. Contour detection using OpenCV
7. Filter contours by minimum area (default: 500px²)

#### Step 2: Corner Detection (corner_detector.py:63-260)

For each detected piece:

1. **Contour Approximation:**
   - Use `cv2.approxPolyDP` with epsilon = 0.012 * perimeter
   - Reduces contour points to key corners

2. **Corner Classification:**
   - **Outer Corners (Convex):** Protruding corners on the piece
   - **Inner Corners (Concave):** Indented corners (puzzle tabs)
   - Uses convexity defects and angle analysis

3. **Straight Edge Detection:** (corner_detector.py:262-326)
   - Analyzes segments between corners
   - Checks if contour points deviate from straight line
   - Uses configurable strictness levels
   - Metrics: median, mean, p90, and max deviation

4. **Frame Corner Detection:** (corner_detector.py:410-631)

   Identifies 90° corners where two straight edges meet (border pieces).

   **Four strict criteria (all must pass):**

   ✓ **Criterion 1 - Connection:** Corner must be between two straight segments

   ✓ **Criterion 2 - Angle:** Angle must be ~90° (within tolerance, default: 85°-95°)

   ✓ **Criterion 3 - Inward Arrow:** Vector pointing toward piece center must fall within the 90° opening angle

   ✓ **Criterion 4 - Convexity:** Corner must be on the outer convex boundary

#### Step 3: Adaptive Strictness (core.py:340-451)

The system automatically adjusts detection sensitivity to find the target number of frame corners:

1. **Start:** Begin with strictest level (`ultra_strict`)
2. **Iterate:** Progressively loosen strictness if too few frame corners found
3. **Goal:** Find at least one frame corner per piece
4. **Target:** Typically 4 frame corners total (one per border piece)

**Strictness Levels (strictest → loosest):**
- `ultra_strict` (0.25x tolerance)
- `ultra_strict_minus` (0.27x) ← **Default optimal**
- `strict_ultra` (0.29x)
- `strict_plus` (0.32x)
- `strict` (0.35x)
- `balanced` (0.5x)
- `loose` (0.75x)
- `ultra_loose` (1.0x)

#### Step 4: SVG Generation

1. **Smoothed SVG** (svg_smoother.py)
   - Creates smooth vector representation of pieces
   - Primary SVG for all further processing

2. **Corners SVG** (svg_corner_drawer.py)
   - Adds visual markers for all detected corners
   - Color-coded: outer (green), inner (blue), frame (red)
   - Two versions: with and without helper lines

#### Step 5: Data Export

- Save analysis data as JSON with piece metadata
- Include corner counts, coordinates, and segment info

### 1.3 Output

**Directory:** `temp/analysis_YYYYMMDD_HHMMSS/`

**Files:**
- `output.png` - Annotated image with green centroid markers
- `pieces_smoothed.svg` - Smoothed piece contours
- `pieces_with_corners_with_helpers.svg` - Corners with helper lines
- `pieces_with_corners_without_helpers.svg` - Clean corner visualization
- `analysis_data.json` - Structured analysis data

**Console Output:**
```
Found 4 pieces
Each piece: outer corners, inner corners, straight edges, frame corners
Final strictness level used
Target frame corners achieved: 4/4
```

---

## Phase 2: Puzzle Solving

**Purpose:** Match puzzle pieces together using geometric analysis

### 2.1 Preparation Sub-Phase (preparation/)

#### Step 1: Load Analysis Data (data_loader.py:32-108)

1. Find most recent analysis folder (or use specified folder)
2. Load SVG file: `pieces_with_corners_without_helpers.svg`
3. Parse SVG to extract piece geometry
4. Convert to `AnalyzedPuzzlePiece` data structures

#### Step 2: Convert SVG to Image (svg_to_image_converter.py)

- Render SVG as raster image
- Fill pieces with orange color for visibility
- Maintain accurate geometric representation

#### Step 3: Segment Contours (contour_segmenter.py:23-64)

For each piece:
1. Extract contour points between consecutive corners
2. Create `ContourSegment` objects with:
   - Start and end corners
   - All contour points along segment
   - Straight/curved classification
   - Border edge flag

#### Step 4: Initial Visualization (visualizer.py)

- Draw all pieces with segments color-coded
- Show corners and frame indicators
- Display legend
- Save as `solver_visualization_output.png`

**Output:** Prepared data structure ready for solving algorithms

### 2.2 Solving Algorithms

Two algorithms available - choose based on your needs:

---

### Algorithm 1: Matrix Solver (matrix_solver/)

**Best for:** Detailed analysis of piece pairs

**Process:** (matrix_solver.py:16-219)

#### Step 1: Select Pieces

- Choose two pieces to compare (default: piece 0 and piece 1)
- Can be specified via parameters

#### Step 2: Generate Similarity Matrices (segment_matcher.py)

For each pair of segments (one from piece 1, one from piece 2):

**A) Length Similarity Matrix**
- Compare segment lengths
- Score: 1.0 - (|len1 - len2| / max(len1, len2))
- Higher score = more similar lengths

**B) Shape Similarity Matrix**
- Normalize segments to same length (100 points)
- Rotate piece 2 segment to find optimal alignment
- Use Iterative Closest Point (ICP) algorithm
- Calculate mean distance between aligned points
- Score based on how well shapes match

**C) Rotation Angle Matrix**
- Store optimal rotation angle found during shape matching
- Used to align pieces correctly

#### Step 3: Extract Diagonal Patterns (diagonal_extractor.py)

- Find diagonal lines in matrices (indicate sequential segment matches)
- Group consecutive matching segments
- Cross-reference between all three matrices

#### Step 4: Validate Groups (group_validator.py)

For each candidate group:
1. Check length similarity (all pairs > threshold)
2. Check shape similarity (all pairs > threshold)
3. Check rotation consistency (angles similar across group)
4. Calculate combined score
5. Rank groups by quality

#### Step 5: Visualization

**Outputs:**
- `segment_match_visualization.png` - Side-by-side pieces with best group highlighted
- `segment_overlay_visualization.png` - Top 3 matches overlaid
- `group_validation_visualization.png` - All validated groups shown

---

### Algorithm 2: Edge Solver (edge_solver/) ⭐ Recommended

**Best for:** Complete puzzle solving with multiple pieces

**Process:** (edge_solver.py:15-191)

#### Step 1: Validate Frame Corners (edge_solver.py:55-68)

- Check all pieces for frame corners
- Only pieces with frame corners can be matched
- Skip interior pieces (no frame corners)
- Need at least 2 pieces with frame corners

#### Step 2: Find Frame-Adjacent Segments (segment_finder.py)

For each piece with frame corners:
1. Identify segments that share a corner with a frame corner
2. These are likely border edges of the puzzle
3. Store frame-adjacent segments for matching

#### Step 3: Match Frame-Adjacent Segments (frame_adjacent_matcher.py)

Compare all pairs of pieces:

**For each segment pair:**

1. **Length Similarity**
   - Compare segment lengths
   - Threshold: lengths must be similar

2. **Shape Similarity**
   - Normalize to same point count
   - Try multiple rotation angles
   - Use ICP for optimal alignment
   - Calculate fit quality

3. **Direction Compatibility**
   - Check if segments face compatible directions
   - Border edges should align properly

4. **Combined Score**
   - Weight: length (30%) + shape (50%) + direction (20%)
   - Sort matches by score

#### Step 4: Calculate Rotation Angles (rotation_calculator.py)

For each piece:
1. Find frame corner coordinates
2. Determine piece orientation based on frame corner position
3. Calculate rotation needed to align with puzzle frame
4. Store rotation angle for final assembly

#### Step 5: Determine Best Connections (connection_manager.py)

For each piece:
1. Find top 2 best matches (each piece connects to 2 others)
2. Avoid duplicate connections
3. Determine which side of piece (left/right based on frame corner)
4. Build connection graph

#### Step 6: Visualize Connections (visualizers.py)

**Interactive Visualization:**
- Show all pieces in a dialog window
- Draw connection lines between matched pieces
- Color-code by connection quality
- Display rotation angles

**SVG Visualization:**
- Generate `edge_solver_connections.svg`
- Vector-based connection diagram
- Clean, scalable visualization

#### Step 7: Build Solution (solution_builder.py)

**Currently:** Creates solution SVG with first piece placed and rotated correctly

**Future Enhancement:** Will assemble all pieces into complete puzzle solution

**Output:** `edge_solver_solution.svg`

---

## Quick Start Guide

### Prerequisites

```bash
# Python 3.8+
# Required packages:
pip install opencv-python numpy matplotlib svgwrite cairosvg
```

### Basic Usage

#### 1. Analyze a Puzzle Image

```python
# Run analysis
python version-v2/analyze.py
```

**Or use as module:**

```python
from puzzle_analyzer_v2.core import analyze_puzzle

# Analyze with defaults
analyzer = analyze_puzzle(
    image_path="../images/puzzle.jpg",
    debug=True
)

# Results saved to: temp/analysis_TIMESTAMP/
```

#### 2. Solve the Puzzle

```python
# Run solver (uses most recent analysis)
python version-v2/solve.py
```

**Or use as module:**

```python
from puzzle_solver_v2.solver import solve_puzzle

# Option 1: Use most recent analysis with edge solver
solve_puzzle(
    solver_algorithm='edge',
    show_visualizations=True
)

# Option 2: Specify analysis folder and pieces (matrix solver)
solve_puzzle(
    temp_folder_name='analysis_20251120_100049',
    piece_id_1=0,
    piece_id_2=1,
    solver_algorithm='matrix',
    show_visualizations=True
)
```

### Advanced Configuration

#### Custom Strictness Level

```python
from puzzle_analyzer_v2.core import analyze_puzzle

# Use stricter detection
analyzer = analyze_puzzle(
    image_path="../images/puzzle.jpg",
    strictness='ultra_strict',  # Fewer straight edges detected
    debug=True,
    target_frame_corners=4  # Expected number of frame corners
)
```

#### Algorithm Selection

```python
from puzzle_solver_v2.solver import solve_puzzle

# Matrix solver - detailed pair analysis
solve_puzzle(
    piece_id_1=0,
    piece_id_2=3,
    solver_algorithm='matrix'
)

# Edge solver - complete puzzle solving
solve_puzzle(
    solver_algorithm='edge'
)
```

---

## Detailed Process Flow

### Complete Pipeline

```
INPUT IMAGE
    ↓
┌─────────────────────────────────────────┐
│  PHASE 1: ANALYSIS                      │
├─────────────────────────────────────────┤
│  1. Image Preprocessing                 │
│     - Grayscale conversion              │
│     - Background detection              │
│     - Thresholding                      │
│                                         │
│  2. Piece Detection                     │
│     - Contour detection                 │
│     - Area filtering                    │
│                                         │
│  3. Corner Analysis                     │
│     - Contour approximation             │
│     - Corner classification             │
│     - Straight edge detection           │
│     - Frame corner detection            │
│                                         │
│  4. Adaptive Strictness                 │
│     - Start strict                      │
│     - Loosen if needed                  │
│     - Achieve target                    │
│                                         │
│  5. SVG Generation                      │
│     - Smooth contours                   │
│     - Mark corners                      │
│     - Export data                       │
└─────────────────────────────────────────┘
    ↓
temp/analysis_TIMESTAMP/
    ↓
┌─────────────────────────────────────────┐
│  PHASE 2: SOLVING                       │
├─────────────────────────────────────────┤
│  PREPARATION:                           │
│  1. Load SVG analysis data              │
│  2. Convert SVG to image                │
│  3. Extract segments                    │
│  4. Create visualization                │
│                                         │
│  SOLVING (Matrix OR Edge):              │
│                                         │
│  Matrix Algorithm:                      │
│    5a. Generate similarity matrices     │
│    6a. Extract diagonal patterns        │
│    7a. Validate groups                  │
│    8a. Visualize matches                │
│                                         │
│  Edge Algorithm:                        │
│    5b. Find frame-adjacent segments     │
│    6b. Match all piece pairs            │
│    7b. Calculate rotations              │
│    8b. Determine connections            │
│    9b. Build solution                   │
└─────────────────────────────────────────┘
    ↓
SOLUTION OUTPUT
```

---

## Algorithm Details

### Frame Corner Detection Algorithm

**Purpose:** Identify 90° corners where puzzle pieces meet the frame

**Specifications:** (corner_detector.py:410-631)

```python
def is_frame_corner(corner, edges, contour, centroid):
    """
    Returns True if corner passes all 4 criteria
    """

    # Criterion 1: Connection
    if not (has_straight_edge_before(corner) and
            has_straight_edge_after(corner)):
        return False

    # Criterion 2: Angle
    angle = calculate_angle(edge_before, edge_after)
    if not (85° <= angle <= 95°):  # Default tolerance
        return False

    # Criterion 3: Inward Arrow
    arrow_to_center = centroid - corner
    if not arrow_falls_within_opening_angle(arrow_to_center,
                                            edge_before,
                                            edge_after):
        return False

    # Criterion 4: Convexity
    if not is_on_convex_hull(corner, contour):
        return False

    return True  # All criteria passed
```

### Segment Matching Algorithm

**Shape Similarity using ICP:**

```python
def calculate_shape_similarity(seg1_points, seg2_points):
    """
    Returns shape similarity score (0-1, higher = better match)
    """

    # 1. Normalize both segments to 100 points
    seg1_norm = interpolate_to_n_points(seg1_points, 100)
    seg2_norm = interpolate_to_n_points(seg2_points, 100)

    # 2. Center both segments at origin
    seg1_centered = seg1_norm - mean(seg1_norm)
    seg2_centered = seg2_norm - mean(seg2_norm)

    # 3. Try multiple rotation angles
    best_score = 0
    best_angle = 0

    for angle in range(0, 360, 5):  # 5-degree increments
        seg2_rotated = rotate(seg2_centered, angle)

        # 4. Apply ICP for fine alignment
        seg2_aligned, transform = icp(seg1_centered, seg2_rotated)

        # 5. Calculate mean distance
        distances = [distance(p1, p2)
                    for p1, p2 in zip(seg1_centered, seg2_aligned)]
        mean_dist = mean(distances)

        # 6. Convert to similarity score
        score = 1.0 / (1.0 + mean_dist)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_score, best_angle
```

### Edge Solver Connection Selection

**For each piece, find best 2 connections:**

```python
def select_best_connections(piece, all_matches):
    """
    Select top 2 connections for a piece
    """

    # 1. Filter matches involving this piece
    piece_matches = [m for m in all_matches
                    if piece in (m.piece1_id, m.piece2_id)]

    # 2. Sort by match score
    piece_matches.sort(key=lambda m: m.match_score, reverse=True)

    # 3. Select top 2 non-overlapping matches
    selected = []
    used_segments = set()

    for match in piece_matches:
        seg_id = get_segment_id_for_piece(match, piece)

        if seg_id not in used_segments:
            selected.append(match)
            used_segments.add(seg_id)

            if len(selected) == 2:
                break

    return selected
```

---

## Configuration Options

### Analyzer Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | required | Path to puzzle image |
| `temp_dir` | str | None | Output directory (auto-generated if None) |
| `strictness` | str | `'ultra_strict_minus'` | Straight edge detection strictness |
| `debug` | bool | False | Enable detailed logging |
| `target_frame_corners` | int | 4 | Expected number of frame corners |

### Solver Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temp_folder_name` | str | None | Analysis folder to use (most recent if None) |
| `piece_id_1` | int | None | First piece for matrix solver |
| `piece_id_2` | int | None | Second piece for matrix solver |
| `solver_algorithm` | str | `'matrix'` | Algorithm: `'matrix'` or `'edge'` |
| `show_visualizations` | bool | True | Display interactive windows |

### Strictness Levels

Lower values = stricter = fewer edges marked as straight = fewer frame corners

| Level | Median | Mean | P90 | Max | Use Case |
|-------|--------|------|-----|-----|----------|
| `ultra_strict` | 0.25x | 0.35x | 0.5x | 1.0x | Nearly perfect lines only |
| `ultra_strict_minus` | 0.27x | 0.38x | 0.55x | 1.1x | **Default - optimal** |
| `strict_ultra` | 0.29x | 0.42x | 0.61x | 1.2x | Very strict |
| `strict_plus` | 0.32x | 0.46x | 0.68x | 1.35x | Strict |
| `strict` | 0.35x | 0.5x | 0.75x | 1.5x | Moderately strict |
| `balanced` | 0.5x | 0.7x | 1.0x | 2.0x | Balanced |
| `loose` | 0.75x | 1.0x | 1.5x | 2.5x | Permissive |
| `ultra_loose` | 1.0x | 1.5x | 2.0x | 3.0x | Very permissive |

---

## Output Files

### Analysis Phase Output

**Directory:** `temp/analysis_YYYYMMDD_HHMMSS/`

#### output.png
- Original image with detected pieces
- Green circles mark centroids
- Blue outlines show piece boundaries
- Piece IDs labeled

#### pieces_smoothed.svg
- Vector representation of smoothed piece contours
- Used as primary input for solver
- Scalable without quality loss

#### pieces_with_corners_with_helpers.svg
- All detected corners marked
- Helper lines show angle measurements
- Color-coded:
  - Green circles: Outer corners
  - Blue circles: Inner corners
  - Red circles with crosshairs: Frame corners

#### pieces_with_corners_without_helpers.svg
- Clean corner visualization
- No helper lines
- Used by solver for data loading

#### analysis_data.json
```json
{
  "timestamp": "2025-11-20T10:00:49",
  "image_path": "../images/puzzle.jpg",
  "image_width": 1920,
  "image_height": 1080,
  "num_pieces": 4,
  "pieces": [
    {
      "id": 0,
      "area": 125430.5,
      "perimeter": 1456.8,
      "centroid": {"x": 450, "y": 380},
      "corners": {
        "total": 8,
        "outer_count": 6,
        "inner_count": 2
      }
    }
  ]
}
```

### Solving Phase Output

**Directory:** `temp/`

#### solver_visualization_output.png
- All pieces with segments drawn
- Color-coded segments (straight vs curved)
- Frame corners highlighted
- Legend included

#### segment_match_visualization.png (Matrix Solver)
- Two pieces side-by-side
- Best matching group highlighted
- Segments numbered

#### segment_overlay_visualization.png (Matrix Solver)
- Top 3 matches overlaid
- Shows how well segments align
- Color-coded by match quality

#### group_validation_visualization.png (Matrix Solver)
- All validated segment groups
- Quality scores displayed
- Helps identify best matches

#### edge_solver_connections.svg (Edge Solver)
- Vector diagram of all connections
- Lines connect matched pieces
- Color intensity = match quality

#### edge_solver_solution.svg (Edge Solver)
- Assembled puzzle solution
- Pieces positioned and rotated
- Final puzzle layout

---

## Troubleshooting

### Analysis Issues

#### No pieces detected
- **Cause:** Low contrast between pieces and background
- **Solution:** Adjust image lighting, increase contrast, or check `min_area` parameter

#### Too many/few frame corners detected
- **Cause:** Incorrect strictness level
- **Solution:**
  - Too many: Use stricter level (`ultra_strict`)
  - Too few: Use looser level (`loose`, `balanced`)
  - Or let adaptive strictness handle it automatically

#### Pieces merged together
- **Cause:** Pieces touching in image
- **Solution:** Ensure pieces are physically separated in the photo

#### Background not detected correctly
- **Cause:** Non-uniform background
- **Solution:** Use uniform background (white or black) with good lighting

### Solving Issues

#### No matches found
- **Cause:** Pieces don't actually connect, or detection quality low
- **Solution:**
  - Verify pieces should connect
  - Re-run analysis with different strictness
  - Check if pieces have frame corners (for edge solver)

#### Poor match quality
- **Cause:** Segment extraction imprecise
- **Solution:**
  - Re-run analysis with `debug=True` to inspect corners
  - Try different strictness level
  - Verify SVG quality

#### Visualizations not showing
- **Cause:** Display backend issue
- **Solution:** Set `show_visualizations=False` and review saved images instead

#### Wrong analysis folder loaded
- **Cause:** Multiple analysis folders exist
- **Solution:** Specify folder explicitly:
  ```python
  solve_puzzle(temp_folder_name='analysis_20251120_100049')
  ```

### Common Error Messages

#### "No analysis folders found"
- **Solution:** Run `analyze.py` first before `solve.py`

#### "Need at least 2 pieces for matching"
- **Solution:** Ensure image contains multiple pieces, check detection parameters

#### "Piece ID X is out of range"
- **Solution:** Use valid piece IDs (0 to N-1), check analysis output for piece count

#### "SVG file not found"
- **Solution:** Analysis phase may have failed, re-run `analyze.py`

---

## Performance Tips

### For Large Images

1. Resize image to reasonable size (1920x1080 is good)
2. Use lower strictness levels for faster processing
3. Disable visualizations: `show_visualizations=False`

### For Many Pieces

1. Use Edge Solver (more efficient for multiple pieces)
2. Process in batches if needed
3. Reduce visualization frequency

### For Best Accuracy

1. High-quality input image (good lighting, focus)
2. Uniform background
3. Physically separated pieces
4. Start with default strictness, let adaptive algorithm work
5. Enable debug mode to verify detection quality

---

## Future Enhancements

### Planned Features

- [ ] Complete puzzle assembly in Edge Solver
- [ ] Interior piece matching (non-border pieces)
- [ ] Multi-piece group assembly
- [ ] Puzzle rotation correction
- [ ] Confidence scoring for final solution
- [ ] Export to physical robot coordinates
- [ ] Real-time solving visualization
- [ ] Machine learning for improved matching

### Contributing

This is a research project. Contributions welcome:
1. Improve matching algorithms
2. Add new visualization types
3. Optimize performance
4. Enhance documentation

---

## Technical Details

### Dependencies

- **OpenCV:** Image processing, contour detection
- **NumPy:** Numerical operations, array manipulation
- **Matplotlib:** Plotting and visualization
- **svgwrite:** SVG generation
- **cairosvg:** SVG to image conversion

### Performance Characteristics

- **Analysis Phase:** ~2-5 seconds for 4 pieces (1920x1080 image)
- **Matrix Solver:** ~1-2 seconds per piece pair
- **Edge Solver:** ~3-5 seconds for 4 pieces with frame corners
- **Memory Usage:** ~200-500 MB depending on image size

### Coordinate System

- Origin: Top-left corner of image
- X-axis: Increases to the right
- Y-axis: Increases downward (standard image coordinates)
- Angles: Measured in degrees, clockwise from positive X-axis

---

## References

### Algorithms Used

- **Contour Detection:** Suzuki85 algorithm (OpenCV implementation)
- **Corner Detection:** Douglas-Peucker approximation + convexity analysis
- **Shape Matching:** Iterative Closest Point (ICP) algorithm
- **Straightness Detection:** Perpendicular distance to line method

### Academic Background

This project builds upon research in:
- Computer vision for puzzle solving
- Geometric shape matching
- Graph-based puzzle assembly
- Jigsaw puzzle solver algorithms

---

## License & Credits

**Version:** 2.0
**Date:** November 2025
**Project:** PREN-ALGO

Developed as part of the PREN puzzle-solving project.

---

## Contact & Support

For questions, issues, or contributions:
- Check existing documentation first
- Review troubleshooting section
- Examine debug output from analysis phase
- Inspect generated visualizations

**Debug Mode:** Always run with `debug=True` when investigating issues for detailed console output.

---

*Last Updated: November 20, 2025*
