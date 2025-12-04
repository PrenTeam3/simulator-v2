"""Convert SVG files to images with custom fill colors."""
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple


class SVGToImageConverter:
    """Converts SVG files to OpenCV images with custom fill colors."""

    @staticmethod
    def convert_svg_to_image(svg_path: Path, fill_color: Tuple[int, int, int] = (255, 165, 0)) -> np.ndarray:
        """
        Convert SVG to OpenCV image with custom fill color for puzzle pieces.

        Args:
            svg_path: Path to SVG file
            fill_color: BGR color tuple for filling pieces (default: orange)

        Returns:
            OpenCV image (numpy array)
        """
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()

        # Get SVG dimensions
        width = int(root.get('width', 800))
        height = int(root.get('height', 600))

        # Create white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Find all path elements with piece data
        namespace = {'svg': 'http://www.w3.org/2000/svg'}
        paths = root.findall('.//svg:path[@data-piece-id]', namespace)
        if not paths:
            # Try without namespace
            paths = [p for p in root.findall('.//path') if p.get('data-piece-id') is not None]

        # Draw each piece
        for path_elem in paths:
            path_d = path_elem.get('d', '')
            if path_d:
                # Parse path and draw filled polygon
                contour_points = SVGToImageConverter._parse_svg_path_to_points(path_d)
                if len(contour_points) > 2:
                    # Convert to numpy array
                    pts = np.array(contour_points, dtype=np.int32)

                    # Fill the polygon with orange color
                    cv2.fillPoly(image, [pts], fill_color)

                    # Draw black outline
                    cv2.polylines(image, [pts], True, (0, 0, 0), 2)

        return image

    @staticmethod
    def _parse_svg_path_to_points(path_d: str) -> list:
        """
        Parse SVG path 'd' attribute to extract points for fillPoly.

        Args:
            path_d: SVG path data string

        Returns:
            List of (x, y) integer coordinate tuples
        """
        import re

        if not path_d:
            return []

        points = []

        # Remove 'Z' (close path) command
        path_d = path_d.replace('Z', '').replace('z', '').strip()

        # Split by command letters
        commands = re.split(r'([MLHVmlhv])', path_d)
        commands = [c.strip() for c in commands if c.strip()]

        current_x, current_y = 0, 0
        i = 0

        while i < len(commands):
            cmd = commands[i]

            if cmd in ['M', 'L'] and i + 1 < len(commands):
                # Absolute move/line
                coords_str = commands[i + 1]
                coord_pairs = coords_str.replace(',', ' ').split()

                j = 0
                while j + 1 < len(coord_pairs):
                    try:
                        x = float(coord_pairs[j])
                        y = float(coord_pairs[j + 1])
                        points.append((int(x), int(y)))
                        current_x, current_y = x, y
                        j += 2
                    except (ValueError, IndexError):
                        j += 1
                i += 2

            elif cmd in ['m', 'l'] and i + 1 < len(commands):
                # Relative move/line
                coords_str = commands[i + 1]
                coord_pairs = coords_str.replace(',', ' ').split()

                j = 0
                while j + 1 < len(coord_pairs):
                    try:
                        dx = float(coord_pairs[j])
                        dy = float(coord_pairs[j + 1])
                        current_x += dx
                        current_y += dy
                        points.append((int(current_x), int(current_y)))
                        j += 2
                    except (ValueError, IndexError):
                        j += 1
                i += 2

            elif cmd == 'H' and i + 1 < len(commands):
                # Absolute horizontal line
                coords_str = commands[i + 1]
                try:
                    x = float(coords_str)
                    current_x = x
                    points.append((int(current_x), int(current_y)))
                except ValueError:
                    pass
                i += 2

            elif cmd == 'h' and i + 1 < len(commands):
                # Relative horizontal line
                coords_str = commands[i + 1]
                try:
                    dx = float(coords_str)
                    current_x += dx
                    points.append((int(current_x), int(current_y)))
                except ValueError:
                    pass
                i += 2

            elif cmd == 'V' and i + 1 < len(commands):
                # Absolute vertical line
                coords_str = commands[i + 1]
                try:
                    y = float(coords_str)
                    current_y = y
                    points.append((int(current_x), int(current_y)))
                except ValueError:
                    pass
                i += 2

            elif cmd == 'v' and i + 1 < len(commands):
                # Relative vertical line
                coords_str = commands[i + 1]
                try:
                    dy = float(coords_str)
                    current_y += dy
                    points.append((int(current_x), int(current_y)))
                except ValueError:
                    pass
                i += 2

            else:
                i += 1

        return points
