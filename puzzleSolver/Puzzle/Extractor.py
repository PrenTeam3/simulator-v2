import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from Img.GreenScreen import remove_background
from Img.filters import export_contours


PREPROCESS_DEBUG_MODE = 0


def show_image(img, ind=None, name="image", show=True):
    """Helper used for matplotlib image display"""
    plt.axis("off")
    plt.imshow(img)
    if show:
        plt.show()


def show_contours(contours, imgRef):
    """Helper used for matplotlib contours display"""
    whiteImg = np.zeros(imgRef.shape)
    cv2.drawContours(whiteImg, contours, -1, (255, 0, 0), 1, maxLevel=1)
    show_image(whiteImg)
    cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "cont.png"), whiteImg)


class Extractor:
    """
    Class used for preprocessing and pieces extraction
    """

    def __init__(self, path, viewer=None, green_screen=False, factor=0.84, black_only=False):
        self.path = path
        self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        self.black_only = black_only
        if green_screen:
            self.img = cv2.medianBlur(self.img, 5)
            divFactor = 1 / (self.img.shape[1] / 640)
            print(self.img.shape)
            print("Resizing with factor", divFactor)
            self.img = cv2.resize(self.img, (0, 0), fx=divFactor, fy=divFactor)
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), self.img)
            remove_background(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "resized.png"), factor=factor)
            self.img_bw = cv2.imread(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "green_background_removed.png"), cv2.IMREAD_GRAYSCALE
            )
            # rescale self.img and self.img_bw to 640
        else:
            self.img_bw = cv2.imread(self.path, cv2.IMREAD_GRAYSCALE)
        self.viewer = viewer
        self.green_ = green_screen
        self.kernel_ = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def log(self, *args):
        """Helper function to log informations to the GUI"""
        print(" ".join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)

    def extract(self):
        """
        Perform the preprocessing of the image and call functions to extract
        informations of the pieces.
        """

        kernel = np.ones((3, 3), np.uint8)

        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png"), self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage("Binarized", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized.png"))

        ### Implementation of random functions, actual preprocessing is down below

        def fill_holes():
            """filling contours found (and thus potentially holes in pieces)"""

            contour, _ = cv2.findContours(
                self.img_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contour:
                cv2.drawContours(self.img_bw, [cnt], 0, 255, -1)

        def generated_preprocesing():
            # For black-only puzzles, use a more lenient threshold or adaptive threshold
            if self.black_only:
                # Try Otsu's adaptive threshold first for better results
                try:
                    threshold_value, self.img_bw = cv2.threshold(
                        self.img_bw, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                    )
                    self.log(f"[BLACK_ONLY] Using Otsu's adaptive threshold: {threshold_value:.1f}")
                except:
                    # Fallback to fixed threshold if Otsu fails
                    threshold_value = 200  # Lower threshold for black pieces
                    self.log(f"[BLACK_ONLY] Otsu failed, using fixed threshold: {threshold_value}")
                    ret, self.img_bw = cv2.threshold(
                        self.img_bw, threshold_value, 255, cv2.THRESH_BINARY_INV
                    )
            else:
                threshold_value = 254  # Original threshold
                ret, self.img_bw = cv2.threshold(
                    self.img_bw, threshold_value, 255, cv2.THRESH_BINARY_INV
                )
            
            cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "otsu_binarized.png"), self.img_bw)
            
            if self.black_only:
                white_pixels = np.sum(self.img_bw == 255)
                total_pixels = self.img_bw.size
                self.log(f"[BLACK_ONLY] Binarization result: {white_pixels} white pixels ({100*white_pixels/total_pixels:.1f}%)")
            
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        def real_preprocessing():
            """Apply morphological operations on base image."""
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_CLOSE, kernel)
            self.img_bw = cv2.morphologyEx(self.img_bw, cv2.MORPH_OPEN, kernel)

        ### PREPROCESSING: starts there

        # With this we apply morphologic operations (CLOSE, OPEN and GRADIENT)
        if not self.green_:
            generated_preprocesing()
        else:
            real_preprocessing()
        # These prints are activated only if the PREPROCESS_DEBUG_MODE variable at the top is set to 1
        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        # With this we fill the holes in every contours, to make sure there is no fragments inside the pieces
        if not self.green_:
            fill_holes()

        if PREPROCESS_DEBUG_MODE == 1:
            show_image(self.img_bw)

        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_treshold_filled.png"), self.img_bw)
        if self.viewer is not None:
            self.viewer.addImage(
                "Binarized treshold", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "binarized_treshold_filled.png")
            )

        contours, hier = cv2.findContours(
            self.img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        self.log("Found nb pieces: " + str(len(contours)))
        
        if self.black_only:
            self.log(f"[BLACK_ONLY] Initial contour count: {len(contours)}")

        # With this we can manually set the maximum number of pieces manually, or we try to guess their number
        # to guess it, we only keep the contours big enough
        nb_pieces = None

        # TEMPORARY TO AVOID DEBUG ORGINAL:
        if len(sys.argv) < 0:
            # Number of pieces specified by user
            nb_pieces = int(sys.argv[2])
            contours = sorted(
                np.array(contours), key=lambda x: x.shape[0], reverse=True
            )[:nb_pieces]
            self.log("Found nb pieces after manual setting: " + str(len(contours)))
        else:
            # Try to remove useless contours
            contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
            
            if self.black_only:
                # For black-only puzzles, be less aggressive with filtering
                # Log contour sizes for debugging
                contour_sizes = [c.shape[0] for c in contours[:10]]  # First 10
                self.log(f"[BLACK_ONLY] Top 10 contour sizes: {contour_sizes}")
                
                if len(contours) >= 2:
                    # Use a more lenient threshold for black-only (1/5 instead of 1/3)
                    max_size = contours[1].shape[0] if len(contours) > 1 else contours[0].shape[0]
                    threshold = max_size / 5.0  # More lenient threshold
                    self.log(f"[BLACK_ONLY] Using threshold: {threshold:.1f} (max_size={max_size:.1f})")
                    original_count = len(contours)
                    contours = [elt for elt in contours if elt.shape[0] > threshold]
                    self.log(f"[BLACK_ONLY] After filtering: {len(contours)} contours (removed {original_count - len(contours)})")
                else:
                    self.log(f"[BLACK_ONLY] Only {len(contours)} contour(s) found, keeping all")
            else:
                # Original filtering for regular puzzles
                if len(contours) >= 2:
                    max_size = contours[1].shape[0]
                    contours = [elt for elt in contours if elt.shape[0] > max_size / 3]
                self.log("Found nb pieces after removing bad ones: " + str(len(contours)))

        if PREPROCESS_DEBUG_MODE == 1:
            show_contours(contours, self.img_bw)  # final contours

        ### PREPROCESSING: the end

        # In case with fail to find the pieces, we fill some holes and then try again
        # while True: # TODO Add this at the end of the project, it is a fallback tactic

        self.log(">>> START contour/corner detection")
        puzzle_pieces = export_contours(
            self.img,
            self.img_bw,
            contours,
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "contours.png"),
            5,
            viewer=self.viewer,
            green=self.green_,
        )
        if puzzle_pieces is None:
            # Export contours error
            return None
        return puzzle_pieces
