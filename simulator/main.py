from __future__ import annotations

"""
PySide6 GUI for visualizing the solver-v2 puzzle solving pipeline.

Usage (from project root):
    python -m simulator.main
"""

import os
import sys
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QTextCursor, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QComboBox,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QSplitter,
    QMessageBox,
    QDialog,
    QScrollArea,
)


# ---------------------------------------------------------------------------
# Wire in solver-v2 modules regardless of current working directory
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_ROOT = PROJECT_ROOT / "solver-v2"

if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

try:
    from analyze import run_analysis  # type: ignore[import]
    from puzzle_solver.solver import solve_puzzle  # type: ignore[import]
except Exception as import_exc:  # noqa: BLE001
    # Defer raising until GUI shows a clear error dialog
    run_analysis = None  # type: ignore[assignment]
    solve_puzzle = None  # type: ignore[assignment]
    _IMPORT_ERROR = import_exc
else:
    _IMPORT_ERROR = None


class PipelineWorker(QThread):
    """
    Background worker that runs the analyze + solve pipeline and reports
    high-level step progress back to the GUI.
    """

    step_started = Signal(str)
    step_finished = Signal(str)
    log_message = Signal(str)
    image_ready = Signal(str, str)  # step_name, image_path
    # temp_folder (str), results (object), error message (str, '' if none)
    pipeline_finished = Signal(str, object, str)

    def __init__(
        self,
        image_path: str,
        solver_algorithm: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._solver_algorithm = solver_algorithm

        # solver-v2 root directory (contains analyze.py, solve.py, temp/, etc.)
        self._v2_root = V2_ROOT

    def _emit_log(self, msg: str) -> None:
        self.log_message.emit(msg)

    def run(self) -> None:
        temp_folder_name: Optional[str] = None
        results: Optional[Dict[str, Any]] = None

        if _IMPORT_ERROR is not None or run_analysis is None or solve_puzzle is None:
            error_msg = f"Import error: {(_IMPORT_ERROR or 'run_analysis/solve_puzzle not available')}"
            self._emit_log(error_msg)
            self.pipeline_finished.emit("", None, error_msg)
            return

        try:
            # Ensure analysis and solver run with solver-v2 as working directory
            os.chdir(self._v2_root)

            # Phase 1: Analysis (covers preprocessing + piece extraction)
            self.step_started.emit("Preprocessing image")
            self._emit_log("Starting analysis phase (preprocessing, detection, corner analysis)...")

            temp_folder_name = run_analysis(  # type: ignore[misc]
                image_path=self._image_path,
                debug=True,
                target_frame_corners=4,
            )

            analysis_dir = self._v2_root / "temp" / temp_folder_name
            threshold_img = analysis_dir / "threshold.png"
            output_img = analysis_dir / "output.png"

            if threshold_img.exists():
                self.image_ready.emit("Preprocessing image", str(threshold_img))

            # Conceptually, analysis also performs piece extraction, so we
            # immediately mark that step as completed as well.
            self.step_finished.emit("Preprocessing image")

            self.step_started.emit("Finding & extracting puzzle pieces")
            self._emit_log("Puzzle pieces detected and analyzed (corners, edges, frame corners).")

            # For UX, reuse the same visualization for this step.
            if output_img.exists():
                self.image_ready.emit("Finding & extracting puzzle pieces", str(output_img))

            self.step_finished.emit("Finding & extracting puzzle pieces")

            # Phase 2: Solving
            self.step_started.emit("Finding matching puzzle pieces")
            self._emit_log(f"Running solver with algorithm='{self._solver_algorithm}'...")

            # Call solver directly to avoid opening its own viewer windows.
            results = solve_puzzle(  # type: ignore[misc]
                temp_folder_name=temp_folder_name,
                piece_id_1=None,
                piece_id_2=None,
                solver_algorithm=self._solver_algorithm,
                show_visualizations=False,
            )

            # Edge-based solvers create visual artifacts; prefer edge_v2 assembly.
            solver_temp_dir = self._v2_root / "temp" / temp_folder_name

            # Try to find some useful solver visualization images.
            candidate_step_images = [
                "solver_visualization_output.png",
                "segment_match_visualization.png",
            ]
            for name in candidate_step_images:
                path = solver_temp_dir / name
                if path.exists() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    self.image_ready.emit("Finding matching puzzle pieces", str(path))
                    break

            self.step_finished.emit("Finding matching puzzle pieces")

            # Assembly step: use final assembled image if available (edge_v2).
            self.step_started.emit("Assembling puzzle")

            assembly_image_path: Optional[Path] = None
            edge_v2_assembly = solver_temp_dir / "assembly_steps_combined.png"
            if edge_v2_assembly.exists():
                assembly_image_path = edge_v2_assembly

            if assembly_image_path is not None:
                self._emit_log(f"Final assembly visualization loaded from: {assembly_image_path}")
                self.image_ready.emit("Assembling puzzle", str(assembly_image_path))
            else:
                self._emit_log("Note: Assembly visualization not found in temp directory.")

            self.step_finished.emit("Assembling puzzle")

            self.pipeline_finished.emit(temp_folder_name, results, "")

        except Exception as exc:  # noqa: BLE001
            error_msg = f"Pipeline error: {exc}\n\n{traceback.format_exc()}"
            self._emit_log(error_msg)
            self.pipeline_finished.emit(temp_folder_name or "", results, error_msg)


class PuzzleSimulatorWindow(QMainWindow):
    """
    Main GUI window for visualizing the puzzle solving pipeline.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Puzzle Solver Simulator")
        self.resize(1200, 800)

        self._project_root = PROJECT_ROOT
        self._v2_root = V2_ROOT
        # Default image in solver-v2/images/
        self._default_image = str(self._project_root / "images" / "puzzle.jpg")

        self._worker: Optional[PipelineWorker] = None

        # State for step-specific visualizations
        self._step_primary_images: Dict[str, str] = {}
        self._matching_detail_images: List[str] = []
        self._assembly_image_path: Optional[str] = None
        self._assembly_frames: List[QPixmap] = []
        self._assembly_index: int = 0
        self._ready_overlay_image: Optional[str] = None  # Track if showing ready overlay

        self._build_ui()

        # Show default image on startup (delayed until window is shown)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._show_default_image_on_startup)

        if _IMPORT_ERROR is not None:
            # Inform user immediately if imports failed
            self._show_startup_import_error(str(_IMPORT_ERROR))

    # --- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # Top control bar: image selection + algorithm + run button
        controls_layout = QHBoxLayout()

        # Image selection
        self.image_label = QLabel(self)
        self.image_label.setText(self._default_image)
        self.image_label.setToolTip("Path to input puzzle image")

        select_image_btn = QPushButton("Select Image...", self)
        select_image_btn.clicked.connect(self._on_select_image_clicked)

        # Algorithm selection
        self.algorithm_combo = QComboBox(self)
        self.algorithm_combo.addItem("Solver v2", userData="edge_v2")
        self.algorithm_combo.setToolTip("Puzzle solving algorithm")

        # Run button
        self.run_button = QPushButton("Run Pipeline", self)
        self.run_button.clicked.connect(self._on_run_clicked)

        controls_layout.addWidget(QLabel("Image:", self))
        controls_layout.addWidget(self.image_label, stretch=1)
        controls_layout.addWidget(select_image_btn)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Algorithm:", self))
        controls_layout.addWidget(self.algorithm_combo)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.run_button)

        main_layout.addLayout(controls_layout)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # Splitter: left (steps + log) / right (image)
        splitter = QSplitter(Qt.Horizontal, self)

        # Left panel
        left_panel = QWidget(self)
        left_layout = QVBoxLayout(left_panel)

        self.steps_list = QListWidget(self)
        self._init_steps()

        left_layout.addWidget(QLabel("Pipeline Steps:", self))
        left_layout.addWidget(self.steps_list, stretch=2)

        left_layout.addWidget(QLabel("Log:", self))
        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text, stretch=3)

        splitter.addWidget(left_panel)

        # Right panel (image view)
        right_panel = QWidget(self)
        right_layout = QVBoxLayout(right_panel)

        right_layout.addWidget(QLabel("Visualization:", self))
        self.image_view = QLabel(self)
        self.image_view.setAlignment(Qt.AlignCenter)
        self.image_view.setMinimumSize(QSize(400, 300))
        self.image_view.setStyleSheet("background-color: #202020; border: 1px solid #404040;")

        right_layout.addWidget(self.image_view, stretch=1)

        # Controls under image (match details, assembly navigation)
        controls_under_image = QHBoxLayout()

        self.match_details_button = QPushButton("More match details...", self)
        self.match_details_button.setEnabled(False)
        self.match_details_button.clicked.connect(self._on_match_details_clicked)

        self.assembly_prev_button = QPushButton("◀ Prev assembly step", self)
        self.assembly_next_button = QPushButton("Next assembly step ▶", self)
        self.assembly_prev_button.setEnabled(False)
        self.assembly_next_button.setEnabled(False)
        self.assembly_prev_button.clicked.connect(self._on_assembly_prev_clicked)
        self.assembly_next_button.clicked.connect(self._on_assembly_next_clicked)

        controls_under_image.addWidget(self.match_details_button)
        controls_under_image.addStretch(1)
        controls_under_image.addWidget(self.assembly_prev_button)
        controls_under_image.addWidget(self.assembly_next_button)

        right_layout.addLayout(controls_under_image)

        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter, stretch=1)

    def _init_steps(self) -> None:
        self.steps_list.clear()
        self._steps_order = [
            "Preprocessing image",
            "Finding & extracting puzzle pieces",
            "Finding matching puzzle pieces",
            "Assembling puzzle",
        ]
        self._step_items: Dict[str, QListWidgetItem] = {}

        for step in self._steps_order:
            item = QListWidgetItem(f"⏺ {step}")
            self.steps_list.addItem(item)
            self._step_items[step] = item

        # React to clicks on steps
        self.steps_list.itemClicked.connect(self._on_step_clicked)

    # --- Helpers ---------------------------------------------------------

    def _set_step_status(self, step_name: str, status: str) -> None:
        """
        Update the visual status for a step.
        status ∈ {"pending", "running", "done"}
        """
        item = self._step_items.get(step_name)
        if item is None:
            return

        prefix = "⏺"
        if status == "running":
            prefix = "▶"
        elif status == "done":
            prefix = "✔"

        item.setText(f"{prefix} {step_name}")

        if status == "running":
            self.steps_list.setCurrentItem(item)

    def _append_log(self, message: str) -> None:
        self.log_text.append(message)
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)

    def _show_image(self, image_path: str) -> None:
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self._append_log(f"Could not load image: {image_path}")
            return
        self.image_view.setPixmap(
            pixmap.scaled(
                self.image_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.image_view.setToolTip(image_path)

    def _show_image_with_ready_overlay(self, image_path: str) -> None:
        """Show an image with 'Ready to analyze' overlay."""
        from pathlib import Path
        img_path = Path(image_path)
        if not img_path.exists():
            self.image_view.setText("Image not found\nClick 'Run Pipeline' to start")
            self._ready_overlay_image = None
            return

        # Load and create overlay with text
        pixmap = QPixmap(str(img_path))
        if pixmap.isNull():
            self.image_view.setText("Click 'Run Pipeline' to start")
            self._ready_overlay_image = None
            return

        # Store that we're showing a ready overlay
        self._ready_overlay_image = image_path

        # Create a painter to add text overlay
        from PySide6.QtGui import QPainter, QFont
        overlay_pixmap = pixmap.copy()
        painter = QPainter(overlay_pixmap)

        # Semi-transparent background for text
        painter.fillRect(0, 0, overlay_pixmap.width(), 150, QColor(0, 0, 0, 180))

        # Draw text
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 24, QFont.Bold)
        painter.setFont(font)
        painter.drawText(20, 50, "Ready to analyze")

        font_small = QFont("Arial", 14)
        painter.setFont(font_small)
        painter.drawText(20, 90, "Click 'Run Pipeline' to start puzzle solving")

        painter.end()

        # Display the image with overlay
        self.image_view.setPixmap(
            overlay_pixmap.scaled(
                self.image_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.image_view.setToolTip(str(img_path))

    def _show_default_image_on_startup(self) -> None:
        """Show the default puzzle image with 'Ready to analyze' overlay."""
        self._show_image_with_ready_overlay(self._default_image)

    def _show_startup_import_error(self, error_text: str) -> None:
        QMessageBox.critical(
            self,
            "Import error",
            "Failed to import solver-v2 modules.\n\n"
            "Ensure you have the following structure relative to the project root:\n"
            "  solver-v2/analyze.py\n"
            "  solver-v2/puzzle_solver/...\n\n"
            f"Underlying error:\n{error_text}",
        )

    # --- Slots / event handlers -----------------------------------------

    def _on_select_image_clicked(self) -> None:
        start_dir = str(Path(self._default_image).parent)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select puzzle image",
            start_dir,
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if file_path:
            self.image_label.setText(file_path)
            # Show the selected image with "Ready to analyze" overlay
            self._show_image_with_ready_overlay(file_path)

    def _on_run_clicked(self) -> None:
        if _IMPORT_ERROR is not None:
            self._show_startup_import_error(str(_IMPORT_ERROR))
            return

        if self._worker is not None and self._worker.isRunning():
            QMessageBox.information(
                self,
                "Pipeline running",
                "The pipeline is already running. Please wait for it to finish.",
            )
            return

        image_path = self.image_label.text().strip()
        if not image_path:
            QMessageBox.warning(self, "No image selected", "Please select an input puzzle image.")
            return

        if not Path(image_path).exists():
            QMessageBox.warning(self, "Image not found", f"Image file does not exist:\n{image_path}")
            return

        algo_code = self.algorithm_combo.currentData()
        if algo_code not in {"edge_v2", "edge", "matrix"}:
            QMessageBox.warning(self, "Invalid algorithm", "Please select a valid algorithm.")
            return

        self._init_steps()
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.image_view.clear()
        self.image_view.setText("Running pipeline...")
        self._step_primary_images.clear()
        self._matching_detail_images = []
        self._assembly_image_path = None
        self._assembly_frames = []
        self._assembly_index = 0
        self._ready_overlay_image = None  # Clear ready overlay when pipeline starts
        self.match_details_button.setEnabled(False)
        self.assembly_prev_button.setEnabled(False)
        self.assembly_next_button.setEnabled(False)

        self.run_button.setEnabled(False)

        self._worker = PipelineWorker(
            image_path=image_path,
            solver_algorithm=algo_code,
            parent=self,
        )

        self._worker.step_started.connect(self._on_step_started)
        self._worker.step_finished.connect(self._on_step_finished)
        self._worker.log_message.connect(self._append_log)
        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.pipeline_finished.connect(self._on_pipeline_finished)

        self._worker.start()

    def _on_step_started(self, step_name: str) -> None:
        self._set_step_status(step_name, "running")

    def _on_step_finished(self, step_name: str) -> None:
        self._set_step_status(step_name, "done")
        # Update progress bar based on completed steps
        done_count = sum(
            1 for step in self._steps_order if self._step_items[step].text().startswith("✔")
        )
        self.progress_bar.setValue(done_count)

    def _on_image_ready(self, step_name: str, image_path: str) -> None:
        self._append_log(f"[{step_name}] Updated visualization from: {image_path}")
        # For the assembly step, we use split sub-images instead of the combined
        # image, so don't automatically display the combined image here.
        if step_name == "Assembling puzzle":
            return

        # Remember latest image for this step
        self._step_primary_images[step_name] = image_path

        # Always show the image when it's ready during pipeline execution
        # (this ensures each step's visualization is displayed as it completes)
        self._show_image(image_path)

    def _on_pipeline_finished(
        self,
        temp_folder_name: str,
        results: Optional[object],
        error: str,
    ) -> None:
        self.run_button.setEnabled(True)

        # Cache paths for step images and details based on the temp folder
        if temp_folder_name:
            base = self._v2_root / "temp" / temp_folder_name

            # Ensure extracting pieces uses output.png
            output_img = base / "output.png"
            if output_img.exists():
                self._step_primary_images.setdefault(
                    "Finding & extracting puzzle pieces", str(output_img)
                )

            # Matching: main view from puzzle_connections_v2.png if present
            connections = base / "puzzle_connections_v2.png"
            if connections.exists():
                # Always prefer puzzle_connections_v2 as the top-level matching image
                self._step_primary_images["Finding matching puzzle pieces"] = str(connections)

            # Collect detailed matching images dynamically
            self._matching_detail_images = []
            # Find all progressive_chain_*.png files
            for path in sorted(base.glob("progressive_chain_*.png")):
                self._matching_detail_images.append(str(path))
            # Find all segment_pairs_*.png files
            for path in sorted(base.glob("segment_pairs_*.png")):
                self._matching_detail_images.append(str(path))

            # Assembly combined image
            assembly = base / "assembly_steps_combined.png"
            if assembly.exists():
                self._assembly_image_path = str(assembly)
                self._build_assembly_frames()

        if error:
            QMessageBox.critical(
                self,
                "Pipeline error",
                f"An error occurred while running the pipeline:\n\n{error}",
            )
        else:
            msg = f"Pipeline completed.\nAnalysis folder: {temp_folder_name}"
            self._append_log(msg)
            QMessageBox.information(self, "Pipeline complete", msg)

        self._worker = None

    # --- Step interactions & extra dialogs -------------------------------

    def _on_step_clicked(self, item: QListWidgetItem) -> None:
        # Item text is like "⏺ Preprocessing image" – strip icon
        text = item.text().lstrip("▶✔⏺ ").strip()

        # Enable/disable extra buttons based on selected step
        if text == "Finding matching puzzle pieces" and self._matching_detail_images:
            self.match_details_button.setEnabled(True)
        else:
            self.match_details_button.setEnabled(False)

        if text == "Assembling puzzle" and self._assembly_frames:
            self.assembly_prev_button.setEnabled(True)
            self.assembly_next_button.setEnabled(True)
        else:
            self.assembly_prev_button.setEnabled(False)
            self.assembly_next_button.setEnabled(False)

        # Choose what to display
        if text == "Preprocessing image":
            # Show threshold image if available, otherwise show original input
            image_path = self._step_primary_images.get(text)
            if image_path:
                self._show_image(image_path)
            else:
                # Fallback to original input image if threshold not yet generated
                fallback_path = self.image_label.text().strip()
                if fallback_path:
                    self._show_image(fallback_path)
            return
        if text == "Assembling puzzle":
            # Start at first sub-picture of the assembly
            if self._assembly_frames:
                self._assembly_index = 0
                self._show_assembly_frame()
            return

        image_path = self._step_primary_images.get(text)
        if image_path:
            self._show_image(image_path)

    def _on_match_details_clicked(self) -> None:
        if not self._matching_detail_images:
            QMessageBox.information(self, "No details", "No detailed matching images found.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Matching details")
        dlg.resize(1400, 900)

        layout = QVBoxLayout(dlg)

        scroll = QScrollArea(dlg)
        scroll.setWidgetResizable(True)
        image_label = QLabel(dlg)
        image_label.setAlignment(Qt.AlignCenter)
        scroll.setWidget(image_label)
        layout.addWidget(scroll, stretch=1)

        buttons_layout = QHBoxLayout()
        prev_btn = QPushButton("◀ Prev", dlg)
        next_btn = QPushButton("Next ▶", dlg)
        zoom_out_btn = QPushButton("Zoom -", dlg)
        zoom_in_btn = QPushButton("Zoom +", dlg)
        buttons_layout.addWidget(prev_btn)
        buttons_layout.addWidget(next_btn)
        buttons_layout.addStretch(1)
        buttons_layout.addWidget(zoom_out_btn)
        buttons_layout.addWidget(zoom_in_btn)
        layout.addLayout(buttons_layout)

        index = {"value": 0}
        zoom = {"factor": 1.0}
        current_pixmap: Dict[str, QPixmap] = {}  # single entry storage

        def update_scaled() -> None:
            if "pixmap" not in current_pixmap:
                return
            base = current_pixmap["pixmap"]
            if base.isNull():
                return
            # Scale relative to original size, keep aspect ratio
            w = int(base.width() * zoom["factor"])
            h = int(base.height() * zoom["factor"])
            if w <= 0 or h <= 0:
                return
            scaled = base.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled)

        def update_image() -> None:
            path = self._matching_detail_images[index["value"]]
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                current_pixmap["pixmap"] = pixmap
                image_label.setToolTip(path)
                update_scaled()
        def on_prev() -> None:
            index["value"] = (index["value"] - 1) % len(self._matching_detail_images)
            update_image()

        def on_next() -> None:
            index["value"] = (index["value"] + 1) % len(self._matching_detail_images)
            update_image()

        prev_btn.clicked.connect(on_prev)
        next_btn.clicked.connect(on_next)

        def on_zoom_in() -> None:
            zoom["factor"] = min(zoom["factor"] * 1.25, 5.0)
            update_scaled()

        def on_zoom_out() -> None:
            zoom["factor"] = max(zoom["factor"] / 1.25, 0.2)
            update_scaled()

        zoom_in_btn.clicked.connect(on_zoom_in)
        zoom_out_btn.clicked.connect(on_zoom_out)

        # Ensure image rescales on dialog resize
        def resize_event(event) -> None:  # type: ignore[override]
            QDialog.resizeEvent(dlg, event)
            # Keep current zoom factor but rescale for new size
            update_scaled()

        dlg.resizeEvent = resize_event  # type: ignore[assignment]

        update_image()
        dlg.exec()

    def _build_assembly_frames(self) -> None:
        """Split the combined assembly image into per-step frames.

        Each step image is stacked vertically and separated by a horizontal
        bar of color #646464 across the full width. We detect these bars and
        cut the image accordingly.
        """
        self._assembly_frames = []
        self._assembly_index = 0
        if not self._assembly_image_path:
            return

        pixmap = QPixmap(self._assembly_image_path)
        if pixmap.isNull():
            return

        width = pixmap.width()
        height = pixmap.height()

        image = pixmap.toImage()
        separator_color = QColor("#646464")

        # Find all horizontal separator rows (gray bar across width)
        separator_rows = []
        sample_xs = [width // 4, width // 2, (3 * width) // 4] if width >= 4 else [0]

        for y in range(height):
            all_gray = True
            for x in sample_xs:
                if image.pixelColor(x, y) != separator_color:
                    all_gray = False
                    break
            if all_gray:
                separator_rows.append(y)

        # Always treat top and bottom as implicit boundaries
        boundaries = [0] + separator_rows + [height]

        for i in range(len(boundaries) - 1):
            y_start = boundaries[i]
            y_end = boundaries[i + 1]
            # Skip pure separator bars or empty ranges
            if y_end - y_start <= 2:
                continue
            frame = pixmap.copy(0, y_start, width, y_end - y_start)
            self._assembly_frames.append(frame)

        # Fallback: if detection failed, keep the whole image as one frame
        if not self._assembly_frames:
            self._assembly_frames.append(pixmap)

    def _show_assembly_frame(self) -> None:
        if not self._assembly_frames:
            return
        frame = self._assembly_frames[self._assembly_index]
        self.image_view.setPixmap(
            frame.scaled(
                self.image_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        if self._assembly_image_path:
            self.image_view.setToolTip(self._assembly_image_path)

    def _on_assembly_prev_clicked(self) -> None:
        if not self._assembly_frames:
            return
        self._assembly_index = (self._assembly_index - 1) % len(self._assembly_frames)
        self._show_assembly_frame()

    def _on_assembly_next_clicked(self) -> None:
        if not self._assembly_frames:
            return
        self._assembly_index = (self._assembly_index + 1) % len(self._assembly_frames)
        self._show_assembly_frame()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        # Keep current image scaled when window is resized
        if self.image_view.pixmap() is not None and self.image_view.toolTip():
            # If showing ready overlay, maintain it during resize
            if self._ready_overlay_image:
                self._show_image_with_ready_overlay(self._ready_overlay_image)
            else:
                self._show_image(self.image_view.toolTip())


def main() -> None:
    app = QApplication(sys.argv)
    window = PuzzleSimulatorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()


