# Puzzle Solver V1 - Documentation

A jigsaw puzzle solving algorithm using computer vision and geometric analysis. This is the first version of the puzzle solver, focusing on detecting puzzle pieces and matching them using matrix-based analysis.

---

## Overview

Version 1 implements a two-phase puzzle solving approach:
1. **Analysis Phase** - Detect and analyze puzzle pieces
2. **Solving Phase** - Match pieces using matrix-based comparison

---

## Quick Start

### Analyze Puzzle

```python
python version-v1/analyze.py
```

### Solve Puzzle

```python
python version-v1/solve.py
```

---

## Architecture

```
version-v1/
├── analyze.py                    # Entry point for analysis
├── solve.py                      # Entry point for solving
│
├── puzzle_analyzer/              # Phase 1: Analysis
│   ├── core.py                   # Main analyzer function
│   ├── geometry.py               # Geometric utilities
│   ├── results_saver.py          # Save results to disk
│   ├── svg_visualizer.py         # SVG generation
│   │
│   └── puzzle_piece/             # Piece analysis modules
│       ├── puzzle_piece_core.py  # PuzzlePiece class
│       ├── corner_detector.py    # Detect corners
│       ├── segment_detector.py   # Detect straight segments
│       ├── border_detector.py    # Identify border edges
│       ├── frame_corner_detector.py  # Frame corner detection
│       ├── curve_fitter.py       # Curve analysis
│       └── visualizer.py         # Draw analysis results
│
└── puzzle_solver/                # Phase 2: Solving
    ├── solver.py                 # Main solver function
    ├── data_loader.py            # Load analysis data
    ├── contour_segmenter.py      # Segment contours
    ├── segment_matcher.py        # Match segments
    ├── diagonal_extractor.py     # Extract diagonal patterns
    ├── group_validator.py        # Validate segment groups
    ├── match_visualizer.py       # Visualize matches
    ├── segment_overlay_visualizer.py  # Overlay segments
    ├── visualizer.py             # General visualization
    ├── image_viewer.py           # Interactive viewer
    └── utils.py                  # Utilities
```

---

## Phase 1: Puzzle Analysis

### Process

1. **Image Preprocessing**
   - Load image
   - Convert to grayscale
   - Gaussian blur for noise reduction
   - Automatic background detection (light/dark)
   - Otsu's thresholding
   - Morphological operations

2. **Piece Detection**
   - Find contours using OpenCV
   - Filter by minimum area (500px²)
   - Create `PuzzlePiece` objects

3. **Piece Analysis** (puzzle_piece_core.py:19-99)
   - **Corner Classification** (corner_detector.py)
     - Outer corners (convex)
     - Inner corners (concave)
     - Curved points
   - **Straight Segment Detection** (segment_detector.py)
     - Identify straight edges between corners
     - Minimum length threshold: 30px
   - **Border Edge Identification** (border_detector.py)
     - Detect potential puzzle frame edges
   - **Frame Corner Detection** (frame_corner_detector.py)
     - Find 90° corners where border edges meet

4. **Visualization & Export**
   - Draw analysis results on image
   - Save to `temp/puzzle_analysis_TIMESTAMP/`
   - Generate SVG vector graphics

### Output Files

**Directory:** `temp/puzzle_analysis_YYYYMMDD_HHMMSS/`

- `analysis_output.png` - Annotated image with all features
- `analysis_data.json` - Structured piece data
- `contours_vector.svg` - Vector graphics representation

### Configuration

Edit `analyze.py`:
```python
image_path = "../images/puzzle.jpg"
show_debug_zones = False    # Show forbidden zones and arrows
verbose_logging = False     # Detailed console output
```

---

## Phase 2: Puzzle Solving

### Process

1. **Load Analysis Data** (data_loader.py)
   - Load most recent or specified analysis folder
   - Parse JSON data into piece objects

2. **Segment Contours** (contour_segmenter.py)
   - Extract segments between corners for each piece
   - Create `ContourSegment` objects

3. **Initial Visualization** (visualizer.py)
   - Draw all pieces with segments
   - Color-code segments
   - Display legend

4. **Segment Matching** (segment_matcher.py)
   - Compare two pieces at a time
   - Generate three matrices:
     - **Length Similarity Matrix** - Compare segment lengths
     - **Shape Similarity Matrix** - ICP-based shape matching
     - **Rotation Angle Matrix** - Optimal rotation angles

5. **Pattern Extraction** (diagonal_extractor.py)
   - Find diagonal patterns in matrices
   - Diagonal patterns indicate sequential matches
   - Group consecutive matching segments

6. **Group Validation** (group_validator.py)
   - Test all identified groups
   - Validate length, shape, and rotation consistency
   - Rank groups by combined score

7. **Visualization**
   - Side-by-side piece comparison with best group highlighted
   - Overlay visualization showing how segments align
   - Group validation visualization

### Output Files

**Directory:** `temp/`

- `solver_visualization_output.png` - All pieces with segments
- `segment_match_visualization.png` - Piece comparison
- `segment_overlay_visualization.png` - Top matches overlaid
- `segment_overlay_specific.png` - Specific segment pairs
- `group_validation_visualization.png` - Group test results

### Configuration

Edit `solve.py`:
```python
# Specify analysis folder and pieces to compare
solve_puzzle(
    temp_folder_name='puzzle_analysis_20251120_102801',
    piece_id_1=0,
    piece_id_2=1
)
```

---

## Key Algorithms

### Frame Corner Detection

Identifies corners where the puzzle piece meets the puzzle frame:

1. Must be at the junction of two border edges
2. Angle between edges ≈ 90° (70°-110° tolerance)
3. Bisector points toward piece center
4. Forbidden zones beyond edges must be clear
5. Must be on convex hull boundary

### Segment Matching

Compares segments from two pieces:

**Length Score:**
```
score = 1.0 - abs(len1 - len2) / max(len1, len2)
```

**Shape Score (ICP):**
1. Normalize segments to 100 points
2. Center both at origin
3. Try multiple rotation angles (0-360°, 5° steps)
4. Apply Iterative Closest Point for alignment
5. Calculate mean distance between aligned points
6. Convert to similarity score: `1.0 / (1.0 + mean_distance)`

**Combined Score:**
```
match_score = 0.3 × length_score + 0.7 × shape_score
```

### Diagonal Extraction

Extracts diagonal lines from match matrices that indicate sequential segment matches:

- Main diagonal: Perfect alignment
- Off-diagonals: Pieces may need rotation or have offset
- Groups consecutive high-scoring cells along diagonals

---

## Usage Examples

### Basic Analysis

```python
from puzzle_analyzer import analyze_puzzle_pieces

pieces = analyze_puzzle_pieces(
    image_path="../images/puzzle.jpg",
    debug_visualization=False,
    verbose_logging=False,
    save_results=True
)
```

### Basic Solving

```python
from puzzle_solver import solve_puzzle

# Use most recent analysis
solve_puzzle()

# Or specify folder and pieces
solve_puzzle(
    temp_folder_name='puzzle_analysis_20251120_102801',
    piece_id_1=0,
    piece_id_2=1
)
```

### Debug Mode

```python
# See forbidden zones, bisector arrows, and detailed logs
pieces = analyze_puzzle_pieces(
    image_path="../images/puzzle.jpg",
    debug_visualization=True,
    verbose_logging=True
)
```

---

## Legend (Visualization)

**Analysis Phase Colors:**
- **Orange:** Straight segments
- **Blue:** Border edges
- **Yellow:** Frame corners (90° corners)
- **Red X:** Inner corners
- **Pink:** Outer corners
- **Orange zones (debug):** Forbidden zones
- **Blue arrow (debug):** Bisector direction
- **Yellow arrow (debug):** Arrow to center

**Solving Phase Colors:**
- Segments color-coded by type
- Best matching group highlighted
- Connection lines show matches

---

## Features

### Analysis Features
- Automatic background detection (light/dark)
- Robust corner classification
- Straight edge detection
- Border edge identification
- Frame corner detection with validation
- SVG vector output
- Interactive zoomable viewer

### Solving Features
- Matrix-based segment matching
- ICP shape alignment
- Diagonal pattern extraction
- Group validation
- Multiple visualization modes
- Side-by-side piece comparison
- Segment overlay visualization

---

## Comparison with V2

| Feature | V1 | V2 |
|---------|----|----|
| Analysis | Basic | SVG-based, Adaptive strictness |
| Solving | Matrix only | Matrix + Edge solver |
| Frame corners | Fixed detection | Adaptive with 4 criteria |
| Architecture | Monolithic | Highly modular |
| Algorithms | 1 (Matrix) | 2 (Matrix + Edge) |
| Output | JSON + PNG | JSON + PNG + SVG |

**V2 Improvements:**
- SVG-based workflow for better accuracy
- Adaptive strictness algorithm
- Edge-based solving for multiple pieces
- Complete puzzle assembly capability
- Better modular architecture
- More configuration options

---

## Troubleshooting

### No pieces detected
- Check image quality and lighting
- Verify pieces are separated
- Ensure sufficient contrast with background

### Incorrect corner detection
- Adjust `min_edge_length` parameter
- Try debug visualization to inspect detection
- Check if background is uniform

### No matches found
- Verify pieces actually connect
- Check segment count (too few/many segments)
- Review analysis quality first

### Analysis folder not found
- Run `analyze.py` before `solve.py`
- Check `temp/` directory exists

---

## Performance

- **Analysis:** ~2-3 seconds for 4 pieces
- **Solving:** ~1-2 seconds per piece pair
- **Memory:** ~200-300 MB

---

## Requirements

```bash
pip install opencv-python numpy matplotlib svgwrite
```

---

## Future Development

Version 1 serves as the foundation. See Version 2 for:
- Enhanced algorithms
- Multiple solving approaches
- Better accuracy
- Complete puzzle assembly
- Modular architecture

---

**Version:** 1.0
**Status:** Stable (Foundation)
**Successor:** See version-v2/ for enhanced implementation

*Last Updated: November 20, 2025*
