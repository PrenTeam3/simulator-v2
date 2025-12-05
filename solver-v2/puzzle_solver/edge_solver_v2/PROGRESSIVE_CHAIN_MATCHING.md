# Progressive Chain Matching Implementation

## Overview
This document describes the progressive chain matching system implemented in the edge_solver_v2 module. The system extends puzzle piece edge matches progressively until they become invalid, allowing us to find the longest valid chain of matching segments.

## Key Concepts

### Chain Extension Philosophy
Instead of limiting chains to a fixed length (e.g., 2 segments), the system:
1. Starts with a valid single segment match (chain length 1)
2. Progressively extends by adding adjacent segments (length 2, 3, 4, ...)
3. Calculates scores after each extension
4. **Stops when**: The chain becomes invalid OR hits a circular boundary
5. **Includes the first invalid match** in results for analysis

### Why Progressive Extension?
- Discovers the maximum valid chain length dynamically
- Shows exactly where and why match quality degrades
- Provides complete visibility into the matching process

## Implementation Details

### Core Components

#### 1. ChainMatcher.extend_match_progressively()
**Location**: `chain_matcher.py` lines 287-413

**Purpose**: Extends a valid segment match progressively until invalid or boundary hit

**Key Features**:
- Returns list of ChainMatch objects for each extension (lengths 1, 2, 3, ...)
- Calculates actual scores for each chain length using `_score_chain()`
- Stops AFTER adding the first invalid match (so you can see why it failed)
- Extensive debug logging at each step

**Stopping Conditions**:
1. Match becomes invalid (fails validation thresholds)
2. Circular boundary hit (segment already in chain)
3. Cannot determine valid extension direction

#### 2. ChainMatcher._score_chain()
**Location**: `chain_matcher.py` lines 125-265

**Purpose**: Calculate length, direction, and shape scores for a complete chain

**Critical Implementation Details**:

##### Chain Alignment (MUST match visualizer exactly)
Before calculating shape similarity, chains must be aligned:

1. **Determine frame endpoints** (lines 200-252):
   - Find frame-touching segments for both pieces
   - Check first segment's neighbors (prev/next)
   - Determine if frame is at start or end of chain
   - Assign B (blue/frame) and R (red/interior) endpoints accordingly

   **IMPORTANT**: This logic MUST be identical to `chain_visualizer.py` lines 287-339, otherwise scores will differ between console and UI!

2. **Translation** (lines 255-257):
   - Translate chain2 so blue (B) endpoints match

3. **Rotation** (lines 259-275):
   - Rotate chain2 around B1 so red (R) endpoints align
   - Also rotate arrow normal vector for direction score

##### Scoring Metrics
- **Length Score**: `100 - (length_diff / max_length * 100)`
- **Direction Score**: `100 - deviation_from_180_degrees`
  - Arrows should point in OPPOSITE directions (180°)
  - Calculated on rotated normal vectors
- **Shape Score**: `100 - (rmsd / avg_length * 100)`
  - RMSD calculated on ALIGNED chains (critical!)
  - Bidirectional distance (min dist from each point to other chain)

##### Validation Thresholds (Chain-specific)
```python
is_valid = (deviation_from_180 < 90.0) and (length_score >= 80.0) and (shape_score >= 80.0)
```

**Note**: Chains use 90° angle tolerance (vs 60° for individual segments) because angle variations accumulate over multiple segments.

#### 3. Progressive Visualization
**Location**: `chain_visualizer.py` lines 13-79

**Purpose**: Display all progressive chain lengths vertically in a single image

**Layout**:
```
┌─────────────────────────────┐
│ Chain Length 1 (VALID)      │
│ [scores displayed]          │
├─────────────────────────────┤
│ Chain Length 2 (VALID)      │
│ [scores displayed]          │
├─────────────────────────────┤
│ Chain Length 3 (INVALID)    │
│ [scores displayed]          │ ← First invalid match, then stops
└─────────────────────────────┘
```

Each row shows:
- Chain overlay with aligned segments
- Length, direction, and shape scores
- "VALID MATCH" (green) or "NOT A MATCH" (red)
- Blue dot = frame connection point
- Red dots = interior endpoints

## Critical Bug Fixes

### Bug #1: Scoring Discrepancy Between Console and UI
**Symptom**: Console showed "Shape score: 0%" but UI showed different score

**Root Cause**:
- `chain_matcher._score_chain()` was comparing UNALIGNED chains
- `chain_visualizer._create_chain_overlay()` was comparing ALIGNED chains
- Different alignments = different RMSD = different scores

**Fix**: Added full alignment logic (translation + rotation) to `_score_chain()` before calculating shape similarity

### Bug #2: Frame Endpoint Determination Mismatch
**Symptom**: Even after alignment fix, scores still differed slightly

**Root Cause**:
- `chain_matcher._score_chain()` used simplified logic: "assume first point = frame"
- `chain_visualizer._create_chain_overlay()` used complex topology analysis to determine which end is frame
- If frame was actually at END of chain, alignment would be backwards

**Fix**: Copied exact frame endpoint determination logic from visualizer to matcher (lines 200-252)

### Bug #3: Artificial Chain Length Limit
**Symptom**: Chains always stopped at length 2

**Root Cause**:
- `edge_solver.py` was passing `max_chain_length=2` parameter
- Old `extend_match_to_chain()` method had `while len(chain) < max_chain_length` logic

**Fix**:
- Removed `max_chain_length` parameter from call
- New `extend_match_progressively()` only stops on invalid match or boundary
- Uses `max_extensions = min(num_segments_p1, num_segments_p2)` as safety limit

## Data Flow

```
valid_matches (from segment matching)
    ↓
ChainMatcher.find_chains_from_matches()
    ↓
For each valid match:
    ├─ extend_match_progressively()
    │   ├─ Add initial match (length 1)
    │   ├─ Determine extension direction (away from frame)
    │   └─ Loop: Extend → Score → Check validity
    │       ├─ If valid: continue to next extension
    │       └─ If invalid: add to list and STOP
    └─ Returns: List[ChainMatch] (progressive extensions)
    ↓
Returns: List[List[ChainMatch]] (one progressive list per initial match)
    ↓
Visualization:
    For each progressive chain list:
        visualize_progressive_chains()
            ├─ Creates overlay for each chain length
            └─ Stacks vertically with scores
```

## File Modifications Summary

### chain_matcher.py
- Added `extend_match_progressively()` - progressive extension logic
- Added `_score_chain()` - scoring with proper alignment
- Modified `find_chains_from_matches()` - returns progressive chain lists
- Removed `max_chain_length` limitation

### chain_visualizer.py
- Added `visualize_progressive_chains()` - stacked vertical display
- Added `_calculate_chain_arrow()` - arrow from entire chain midpoint
- Modified `_create_chain_overlay()` - added scoring metrics display
- Increased angle tolerance to 90° for chain validation

### edge_solver.py
- Modified chain visualization loop to handle progressive chains
- Removed `max_chain_length=2` parameter
- Changed output filename to `progressive_chain_P{id1}_P{id2}.png`

### visualizer.py
- Added delegation method `visualize_progressive_chains()`

## Testing & Verification

### Expected Behavior
1. Console logs should show:
   ```
   Extension step 1 (attempting chain length 2):
   Scores: Length=95.2%, Direction=88.3%, Shape=82.1%
   Valid: True
   Match is VALID - continuing to next extension

   Extension step 2 (attempting chain length 3):
   Scores: Length=91.5%, Direction=85.7%, Shape=78.3%
   Valid: False
   Match became INVALID - stopping (included this invalid match)

   Completed: Generated 3 progressive chain(s) (lengths: [1, 2, 3])
   ```

2. UI should display 3 rows (lengths 1, 2, 3) with SAME scores as console

3. Chain should stop at first invalid match and include it in visualization

### Common Issues

**Scores differ between console and UI**:
- Check that alignment logic is IDENTICAL in both places
- Verify frame endpoint determination matches exactly

**Chain stops at length 2**:
- Check for any remaining `max_chain_length` parameters
- Verify `extend_match_progressively()` is being called (not old method)

**Chain extends forever**:
- Check validation thresholds (90° angle, 80% length/shape)
- Verify circular boundary detection is working

## Future Improvements

1. **Adaptive thresholds**: Adjust validation thresholds based on chain length
2. **Performance optimization**: Cache alignment calculations
3. **Better stopping criteria**: Consider rate of score degradation
4. **Multi-path exploration**: Try both extension directions simultaneously

## Key Takeaways

1. **Always align before comparing**: Shape similarity requires aligned chains
2. **Match the visualizer exactly**: Any difference in logic causes score discrepancies
3. **Progressive is better than fixed**: Discovers natural chain boundaries
4. **Include the failure**: First invalid match shows why extension stopped
5. **Debug extensively**: Detailed logging is essential for understanding behavior
