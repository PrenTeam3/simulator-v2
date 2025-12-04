# Puzzle Solver Simulator GUI

A graphical interface for visualizing the puzzle solving pipeline with step-by-step visualization.

## Installation

1. **Install Python dependencies:**

   ```bash
   pip install -r simulator\requirements.txt
   ```

   This installs:
   - OpenCV (image processing)
   - NumPy (numerical operations)
   - SciPy (scientific computing)
   - PySide6 (GUI framework)

## Usage

### Running the Simulator

From the project root directory:

```bash
python -m simulator.main
```

### Basic Workflow

1. **Select Input Image:**
   - Click "Select Image..." to choose a puzzle image file
   - Default path: `solver-v2/images/puzzle.jpg`

2. **Choose Algorithm:**
   - **Edge Solver V2** (recommended) - Advanced edge-based matching with assembly
   - **Edge Solver V1** - Original edge-based matching, doesnt really work
   - **Matrix Solver** - Matrix-based matching approach, doesnt really work

3. **Run Pipeline:**
   - Click "Run Pipeline" to start analysis and solving
   - Progress is shown in the progress bar and step list
   - Log messages appear in the log panel

4. **View Step Visualizations:**
   - Click any step in the "Pipeline Steps" list to view its visualization:
     - **Preprocessing image** - Shows the input puzzle image
     - **Finding & extracting puzzle pieces** - Shows detected pieces (`output.png`)
     - **Finding matching puzzle pieces** - Shows connection diagram (`puzzle_connections_v2.png`)
       - Click "More match details..." to browse detailed matching visualizations
     - **Assembling puzzle** - Shows assembly steps (navigate with Prev/Next buttons)
