# Edge Solver Module Structure

This module has been refactored from a single 1494-line file into a modular structure for better maintainability.

## Module Organization

### Main Entry Point
- **edge_solver.py** (191 lines) - Main solver class that orchestrates the edge-based solving process

### Core Logic Modules
- **segment_finder.py** (67 lines) - Finding frame-adjacent segments
- **frame_adjacent_matcher.py** (73 lines) - Matching segments between pieces
- **connection_manager.py** (60 lines) - Determining best connections between pieces
- **rotation_calculator.py** (117 lines) - Calculating piece rotation angles

### Visualization & Output
- **visualizers.py** (327 lines) - All visualization functions (SVG and interactive)
- **solution_builder.py** (469 lines) - Building and assembling puzzle solutions

## Benefits of Refactoring

1. **Better Maintainability**: Each module has a single, clear responsibility
2. **Easier Testing**: Individual components can be tested in isolation
3. **Improved Code Navigation**: Smaller files are easier to understand and navigate
4. **Clearer Dependencies**: Import structure makes dependencies explicit
5. **Reduced Complexity**: Main solver file is now ~13% of original size (191 vs 1494 lines)

## Usage

The public API remains unchanged. Import and use EdgeSolver as before:

```python
from puzzle_solver_v2.edge_solver import EdgeSolver

result = EdgeSolver.solve_with_edges(prepared_data, show_visualizations=True)
```

All other classes are also exported for advanced usage if needed.
