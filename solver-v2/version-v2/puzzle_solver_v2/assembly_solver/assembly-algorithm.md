# Assembly Solver Algorithm

## Overview
Places 4 puzzle pieces together using chain connections from the edge solver.

## Algorithm Steps

### 1. Calculate Centroids & Chain Endpoints
- Extract centroids from all pieces
- Get pre-calculated red/blue dots from chain matching
- Store connection data for each piece

### 2. Normalize Orientations
- Rotate each piece to align frame edges with X/Y axes
- Find two straight edges adjacent to frame corner
- Apply minimal rotation to achieve 0° (horizontal) and -90° (vertical) alignment

### 3. Select Anchor Piece
- Score pieces based on total connection quality
- Select corner piece with highest connection scores
- This piece is placed first

### 4. Position Anchor
- Place anchor piece at origin (100, 100)
- Test rotations (0°, 90°, 180°, 270°) to ensure frame corner is at top-left
- Apply final translation and rotation

### 5. Place Second Piece
- Find highest-scoring connection to anchor
- **Chain alignment**: Translate B2→B1, then rotate around B1 to align R2→R1
- Store placed piece with final position and rotation

### 6. Place Third Piece
- Find second connection to anchor
- Apply same chain alignment algorithm

### 7. Place Fourth Piece
- **Tree approach**: Connect to child pieces (2 or 3), not anchor
- Apply same chain alignment algorithm

### 8. Final Assembly
- Return all placed pieces with positions and rotations

## Key Transformation Pipeline
1. **Normalization rotation** - Align to axes
2. **Placement rotation** - Correct orientation (anchor only)
3. **Translation** - Move to target position
4. **Alignment rotation** - Match chains using red/blue dots

## Core Alignment Algorithm
For each new piece:
1. Transform red/blue dots from already-placed piece to final position
2. Normalize red/blue dots of new piece (axis alignment only)
3. Translate new piece so B2 = B1
4. Rotate new piece around B1 so R2 = R1
