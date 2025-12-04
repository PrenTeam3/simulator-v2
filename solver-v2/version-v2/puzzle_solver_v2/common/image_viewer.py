"""Interactive image viewer with zoom and pan capabilities."""
import cv2
import numpy as np


class InteractiveImageViewer:
    """Display image with zoom and pan controls."""

    def __init__(self, window_name: str = "Image Viewer"):
        """Initialize viewer with window name."""
        self.window_name = window_name
        self.original_image = None
        self.current_image = None
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.mouse_down = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zoom and pan."""
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with mouse wheel, centered on cursor
            old_zoom = self.zoom_level
            if flags > 0:  # Scroll up - zoom in
                self.zoom_level *= 1.1
            else:  # Scroll down - zoom out
                self.zoom_level /= 1.1

            self.zoom_level = max(0.1, min(self.zoom_level, 5.0))  # Clamp zoom

            # Adjust pan to keep cursor position fixed
            # The mouse is at (x, y) in the window, which corresponds to (pan_x + x, pan_y + y) in the zoomed image
            # After zoom, we want the same image point to be at (x, y) in the window
            zoom_ratio = self.zoom_level / old_zoom
            self.pan_x = int((self.pan_x + x) * zoom_ratio - x)
            self.pan_y = int((self.pan_y + y) * zoom_ratio - y)

            self.update_display()

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
            self.last_mouse_x = x
            self.last_mouse_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False

        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_down:
            # Pan with mouse drag
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            self.pan_x -= dx
            self.pan_y -= dy
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.update_display()

    def update_display(self):
        """Update the displayed image based on zoom and pan."""
        if self.original_image is None:
            return

        h, w = self.original_image.shape[:2]
        scaled_h = int(h * self.zoom_level)
        scaled_w = int(w * self.zoom_level)

        # Resize image
        if self.zoom_level != 1.0:
            resized = cv2.resize(self.original_image, (scaled_w, scaled_h))
        else:
            resized = self.original_image.copy()

        # Get window size
        window_h = 800
        window_w = 1200

        # Calculate valid pan bounds
        max_pan_x = max(0, scaled_w - window_w)
        max_pan_y = max(0, scaled_h - window_h)

        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

        # Crop the visible region
        visible = resized[
            self.pan_y : self.pan_y + window_h,
            self.pan_x : self.pan_x + window_w
        ]

        # Pad if image is smaller than window
        if visible.shape[0] < window_h or visible.shape[1] < window_w:
            padded = np.ones((window_h, window_w, 3), dtype=np.uint8) * 40
            padded[: visible.shape[0], : visible.shape[1]] = visible
            visible = padded

        cv2.imshow(self.window_name, visible)

    def show(self, image: np.ndarray, start_at_bottom: bool = False, initial_zoom: float = 1.0):
        """Display image with interactive controls.

        Args:
            image: Image to display
            start_at_bottom: If True, start scrolled to the bottom of the image
            initial_zoom: Initial zoom level (default 1.0, use values like 0.5 for zoomed out)
        """
        self.original_image = image.copy()
        self.current_image = image.copy()
        self.zoom_level = initial_zoom
        self.pan_x = 0

        # If start_at_bottom is True, set pan_y to show the bottom of the image
        if start_at_bottom:
            h, w = self.original_image.shape[:2]
            window_h = 800
            # Calculate pan_y based on the zoomed size
            scaled_h = int(h * self.zoom_level)
            # Set pan_y to maximum (bottom of image)
            self.pan_y = max(0, scaled_h - window_h)
        else:
            self.pan_y = 0

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.update_display()

        print("\n" + "="*70)
        print("IMAGE VIEWER CONTROLS")
        print("="*70)
        print("Mouse Wheel:  Zoom in/out")
        print("Mouse Drag:   Pan around image")
        print("Press any key to close")
        print("="*70 + "\n")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
