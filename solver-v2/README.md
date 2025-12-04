# PREN-ALGO - Jigsaw Puzzle Solver

Automated jigsaw puzzle solving using computer vision and geometric analysis. This repository contains two versions of the puzzle solver, each with different approaches and capabilities.

---

## Overview

This project develops algorithms to automatically solve jigsaw puzzles by:
1. Detecting puzzle pieces from images
2. Analyzing geometric features (corners, edges, frame corners)
3. Matching pieces together using sophisticated algorithms
4. Assembling the complete puzzle solution

**Project:** PREN Team 3
**Purpose:** Automated puzzle solving for robotics applications

---

## Quick Navigation

| Version | Status | Best For | Documentation |
|---------|--------|----------|---------------|
| [Version 1](version-v1/) | Stable (Foundation) | Learning, Simple puzzles | [V1 README](version-v1/README.md) |
| [Version 2](version-v2/) | Current (Enhanced) | Production, Complex puzzles | [V2 README](version-v2/README.md) |

---

## Version Comparison

### Version 1 - Foundation

**Location:** `version-v1/`

Basic implementation with matrix-based solving:

**Features:**
- Piece detection with contour analysis
- Corner classification (outer/inner)
- Straight edge and border detection
- Frame corner detection
- Matrix-based segment matching
- Pair-wise piece comparison

**Best for:**
- Understanding the fundamentals
- Simple puzzle problems
- Educational purposes
- Proof of concept

**Algorithms:** 1 (Matrix-based)

[📖 View V1 Documentation](version-v1/README.md)

---

### Version 2 - Enhanced ⭐ Recommended

**Location:** `version-v2/`

Advanced implementation with multiple solving strategies:

**Features:**
- SVG-based analysis for precision
- Adaptive strictness algorithm
- Enhanced frame corner detection (4 strict criteria)
- Two solving algorithms:
  - Matrix solver (detailed pair analysis)
  - Edge solver (multi-piece assembly)
- Complete puzzle assembly
- Highly modular architecture
- Extensive visualization tools

**Best for:**
- Production use
- Complex puzzles
- Multi-piece solving
- Research and development

**Algorithms:** 2 (Matrix + Edge-based)

[📖 View V2 Documentation](version-v2/README.md)

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd PREN-ALGO

# Install dependencies
pip install opencv-python numpy matplotlib svgwrite cairosvg
```

### Running Version 2 (Recommended)

```bash
# 1. Analyze puzzle image
python version-v2/analyze.py

# 2. Solve the puzzle
python version-v2/solve.py
```

### Running Version 1

```bash
# 1. Analyze puzzle image
python version-v1/analyze.py

# 2. Solve the puzzle
python version-v1/solve.py
```

---

## Key Differences

| Feature | Version 1 | Version 2 |
|---------|-----------|-----------|
| **Analysis Method** | OpenCV contours | SVG-based geometry |
| **Strictness** | Fixed parameters | Adaptive algorithm |
| **Frame Corner Detection** | Basic validation | 4-criteria specification |
| **Solving Algorithms** | Matrix only | Matrix + Edge |
| **Multi-piece Solving** | No | Yes (Edge solver) |
| **Puzzle Assembly** | Pair-wise only | Complete assembly |
| **Architecture** | Monolithic | Highly modular |
| **Code Organization** | Mixed | Separated phases |
| **Visualization** | Basic | Comprehensive |
| **Output Formats** | JSON + PNG | JSON + PNG + SVG |
| **Rotation Calculation** | Per match | Per piece + global |
| **Connection Management** | Manual | Automated graph |

---

## Project Structure

```
PREN-ALGO/
├── README.md                 # This file
├── images/                   # Input puzzle images
│   └── puzzle.jpg
│
├── version-v1/              # Version 1 (Foundation)
│   ├── README.md            # V1 documentation
│   ├── analyze.py           # Analysis entry point
│   ├── solve.py             # Solving entry point
│   ├── puzzle_analyzer/     # Analysis modules
│   └── puzzle_solver/       # Solving modules
│
├── version-v2/              # Version 2 (Enhanced) ⭐
│   ├── README.md            # V2 documentation
│   ├── analyze.py           # Analysis entry point
│   ├── solve.py             # Solving entry point
│   ├── puzzle_analyzer_v2/  # Analysis modules
│   └── puzzle_solver_v2/    # Solving modules
│       ├── preparation/     # Data loading
│       ├── matrix_solver/   # Matrix algorithm
│       ├── edge_solver/     # Edge algorithm
│       └── common/          # Shared utilities
│
└── temp/                    # Output directory (auto-generated)
    ├── puzzle_analysis_*/   # V1 analysis results
    └── analysis_*/          # V2 analysis results
```

---

## Workflow

Both versions follow a similar two-phase approach:

### Phase 1: Analysis

```
INPUT IMAGE
    ↓
Preprocessing (grayscale, threshold, morphology)
    ↓
Piece Detection (contour finding)
    ↓
Corner Analysis (classify corners)
    ↓
Edge Detection (straight segments)
    ↓
Frame Corner Detection (border pieces)
    ↓
OUTPUT: Analysis data + Visualizations
```

### Phase 2: Solving

**Version 1:**
```
Load Analysis Data
    ↓
Segment Contours
    ↓
Match Segments (Matrix-based)
    ↓
Extract Diagonal Patterns
    ↓
Validate Groups
    ↓
OUTPUT: Match visualizations
```

**Version 2:**
```
Load Analysis Data
    ↓
Segment Contours
    ↓
Choose Algorithm:
    ├─→ Matrix Solver (pair analysis)
    └─→ Edge Solver (multi-piece)
        ↓
    Find Frame-Adjacent Segments
        ↓
    Match All Piece Pairs
        ↓
    Calculate Rotations
        ↓
    Determine Connections
        ↓
    Build Solution
        ↓
OUTPUT: Complete puzzle solution
```

---

## Key Algorithms

### Frame Corner Detection

Both versions detect 90° corners where puzzle pieces meet the frame:

**V1 Criteria:**
- Junction of two border edges
- ~90° angle
- Bisector toward center
- Clear forbidden zones
- On convex hull

**V2 Enhanced Criteria:**
1. **Connection:** Two straight edges meet
2. **Angle:** 90° ± 5° (strict)
3. **Inward Arrow:** Points within opening angle
4. **Convexity:** On outer boundary

V2 adds adaptive strictness to find optimal detection level.

### Segment Matching

Both versions use similar matching logic:

1. **Length Similarity:** Compare segment lengths
2. **Shape Similarity:** ICP-based alignment
3. **Rotation Angle:** Optimal rotation for match
4. **Combined Score:** Weighted combination

**V2 Improvements:**
- Better ICP implementation
- More robust normalization
- Direction compatibility checking
- Multi-piece graph building

---

## Input Requirements

**Image Format:** JPG, PNG, or common image formats

**Image Quality:**
- Good lighting (uniform, no harsh shadows)
- Clear background (white or black preferred)
- Pieces separated (not touching)
- Minimal blur (sharp focus)
- Resolution: 1920x1080 or similar (adjustable)

**Puzzle Requirements:**
- Standard jigsaw pieces
- Clear edges and corners
- At least 4 pieces (including border pieces)
- Border pieces identifiable by straight edges

---

## Output Files

### Version 1 Output

**Directory:** `temp/puzzle_analysis_TIMESTAMP/`
- `analysis_output.png` - Annotated image
- `analysis_data.json` - Piece data
- `contours_vector.svg` - Vector graphics
- `solver_visualization_output.png` - Match results

### Version 2 Output

**Directory:** `temp/analysis_TIMESTAMP/`
- `output.png` - Annotated image
- `pieces_smoothed.svg` - Smoothed contours
- `pieces_with_corners_*.svg` - Corner visualizations
- `analysis_data.json` - Structured data
- `edge_solver_solution.svg` - Complete solution (Edge solver)

---

## Development Timeline

**Version 1 (Foundation)**
- Initial implementation
- Basic algorithms established
- Proof of concept successful
- Matrix-based matching working

**Version 2 (Enhancement)**
- SVG-based analysis pipeline
- Adaptive strictness algorithm
- Edge solver implementation
- Modular architecture refactoring
- Edge solver split into 6 modules (Nov 2025)
- Complete puzzle assembly capability

---

## Use Cases

### Research & Development
- Algorithm development and testing
- Computer vision experiments
- Geometric analysis studies

### Educational
- Learning puzzle solving algorithms
- Understanding computer vision
- Graph theory applications

### Robotics Applications
- Automated puzzle assembly
- Vision-guided manipulation
- Task planning and coordination

### Production (V2)
- Reliable puzzle solving
- Multiple piece configurations
- Complete assembly solutions

---

## Performance

| Metric | Version 1 | Version 2 |
|--------|-----------|-----------|
| Analysis Time (4 pieces) | ~2-3 sec | ~2-5 sec |
| Solving Time (pair) | ~1-2 sec | ~1-2 sec |
| Solving Time (all) | N/A | ~3-5 sec |
| Memory Usage | ~200-300 MB | ~200-500 MB |
| Accuracy | Good | Excellent |
| Scalability | Limited | Better |

---

## Getting Help

1. **Check version-specific README:**
   - [Version 1 Documentation](version-v1/README.md)
   - [Version 2 Documentation](version-v2/README.md)

2. **Enable debug mode:**
   ```python
   # V1
   analyze_puzzle_pieces(image_path, debug_visualization=True, verbose_logging=True)

   # V2
   analyze_puzzle(image_path, debug=True)
   ```

3. **Review output visualizations:**
   - Check `temp/` directory for saved images
   - Inspect corner detection quality
   - Verify segment extraction

4. **Common issues:**
   - Poor lighting → Improve image quality
   - No pieces detected → Check background contrast
   - No matches found → Verify piece connectivity
   - Frame corners missing → Try different strictness (V2)

---

## Technical Stack

**Core Technologies:**
- **OpenCV:** Image processing and computer vision
- **NumPy:** Numerical computations
- **Matplotlib:** Visualization and plotting
- **svgwrite:** SVG generation (V2)
- **cairosvg:** SVG rendering (V2)

**Algorithms:**
- Contour detection (Suzuki85)
- Douglas-Peucker approximation
- Iterative Closest Point (ICP)
- Convexity analysis
- Geometric transformations
- Graph-based assembly (V2)

---

## Future Work

- [ ] Machine learning for improved matching
- [ ] Real-time solving visualization
- [ ] 3D puzzle support
- [ ] Texture-based matching
- [ ] Confidence scoring
- [ ] Robot integration and control
- [ ] Web-based interface
- [ ] Parallel processing optimization
- [ ] Support for irregular puzzles
- [ ] Piece rotation prediction

---

## Contributing

This is an active research project. Areas for contribution:
- Algorithm improvements
- Performance optimization
- Additional solving strategies
- Enhanced visualizations
- Documentation improvements
- Bug fixes and testing

---

## License & Credits

**Project:** PREN Team 3
**Purpose:** Automated puzzle solving
**Date:** 2025

Developed for robotics applications in puzzle assembly and manipulation.

---

## Version History

| Version | Release | Status | Notes |
|---------|---------|--------|-------|
| 1.0 | 2025 | Stable | Foundation implementation |
| 2.0 | 2025 | Current | Enhanced with dual algorithms |

---

## Recommended Path

**New Users:** Start with Version 2 for best results

**Learning:** Review Version 1 for fundamentals, then explore Version 2 enhancements

**Production:** Use Version 2 with Edge solver

**Development:** Build on Version 2's modular architecture

---

*For detailed documentation, see version-specific README files.*

**Last Updated:** November 20, 2025