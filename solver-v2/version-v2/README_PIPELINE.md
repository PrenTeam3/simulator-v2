# Version-v2 Pipeline Usage Guide

This document explains how to use the combined analyze + solve pipeline.

## Overview

The version-v2 pipeline has been updated to support three modes of operation:

1. **Individual Analysis** - Run only the analysis phase
2. **Individual Solving** - Run only the solving phase (using existing analysis)
3. **Combined Pipeline** - Run both analysis and solving in sequence

## Files

- **`analyze.py`** - Analyzes puzzle image and saves results to `temp/analysis_TIMESTAMP/`
- **`solve.py`** - Solves puzzle using existing analysis data
- **`run.py`** - NEW: Combines both analyze and solve in one run

## Usage

### Option 1: Combined Pipeline (Recommended for New Images)

Run both analysis and solving in one command:

```bash
python run.py
```

The default configuration in `run.py`:
```python
temp_folder, results = run_full_pipeline(
    image_path="../images/puzzle.jpg",
    solver_algorithm='edge_v2',
    show_visualizations=True
)
```

**Customization options:**
```python
# With specific piece comparison
temp_folder, results = run_full_pipeline(
    image_path="../images/puzzle.jpg",
    solver_algorithm='edge_v2',
    piece_id_1=1,
    piece_id_2=3,
    show_visualizations=False,
    debug_analysis=True
)

# Different image and algorithm
temp_folder, results = run_full_pipeline(
    image_path="../images/puzzle5.png",
    solver_algorithm='matrix',
    target_frame_corners=4
)
```

### Option 2: Individual Analysis

Run only the analysis phase:

```bash
python analyze.py
```

This will:
- Analyze the puzzle image
- Save results to `temp/analysis_TIMESTAMP/`
- Print the folder name for use with solve.py

**Example output:**
```
======================================================================
Analysis complete! Results saved to: analysis_20251126_143022
======================================================================

You can now run solve.py with folder: analysis_20251126_143022
```

### Option 3: Individual Solving

Run only the solving phase (using existing analysis):

```bash
python solve.py
```

By default, it uses the most recent analysis. To specify a folder:

```python
# In solve.py, uncomment and modify:
run_solver(temp_folder_name='analysis_20251120_100049')
```

### Option 4: Import as Module

You can also import the functions in your own scripts:

```python
from analyze import run_analysis
from solve import run_solver

# Run analysis first
temp_folder = run_analysis(
    image_path="../images/puzzle.jpg",
    debug=False,
    target_frame_corners=4
)

# Then solve using the results
results = run_solver(
    temp_folder_name=temp_folder,
    solver_algorithm='edge_v2',
    show_visualizations=True
)
```

## Parameters

### `run_analysis()` parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_path` | str | `"../images/puzzle.jpg"` | Path to puzzle image |
| `debug` | bool | `True` | Enable debug logging |
| `target_frame_corners` | int | `4` | Target number of frame corners to find |

**Returns:** `str` - Name of the temp folder (e.g., `'analysis_20251120_100049'`)

### `run_solver()` parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temp_folder_name` | str | `None` | Name of analysis folder (if None, uses most recent) |
| `piece_id_1` | int | `None` | First piece ID for comparison (optional) |
| `piece_id_2` | int | `None` | Second piece ID for comparison (optional) |
| `solver_algorithm` | str | `'edge_v2'` | Algorithm: `'matrix'`, `'edge'`, or `'edge_v2'` |
| `show_visualizations` | bool | `True` | Whether to display visualizations |

**Returns:** `dict` - Solver results

### `run_full_pipeline()` parameters:

Combines all parameters from both functions above.

## Algorithm Options

Three solver algorithms are available:

1. **`'matrix'`** - Matrix-based matching approach
2. **`'edge'`** - Edge-based matching (v1)
3. **`'edge_v2'`** - Advanced edge-based matching with assembly (recommended)

## Folder Structure

```
version-v2/
├── analyze.py          # Analysis entry point (can be imported)
├── solve.py            # Solving entry point (can be imported)
├── run.py              # Combined pipeline (NEW)
├── temp/
│   ├── analysis_20251120_100049/   # Analysis results with timestamp
│   │   ├── pieces_smoothed.svg
│   │   ├── pieces_with_corners.svg
│   │   ├── analysis_data.json
│   │   └── output.png
│   └── ...
└── ...
```

## Backward Compatibility

All existing scripts and workflows continue to work:
- Running `python analyze.py` directly still works
- Running `python solve.py` directly still works
- Existing imports remain functional

The new `run_analysis()` and `run_solver()` functions are additions that enhance functionality without breaking existing code.
