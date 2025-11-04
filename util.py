import numpy as np
import cv2

def get_limits(color):
        # Ensure the color array is uint8 so OpenCV can convert color spaces correctly
        c = np.uint8([[color]])
        hsv_c = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

        # Hue in OpenCV HSV is [0, 179]. Clamp to avoid wrap-around when casting.
        h = int(hsv_c[0, 0, 0])
        lower_h = max(h - 10, 0)
        upper_h = min(h + 10, 179)

        # Tighten saturation and value thresholds to reduce false positives
        # Higher S (saturation) = more vivid color (not washed out)
        # Higher V (value) = brighter color (not dim)
        lowerLimit = np.array([lower_h, 150, 150], dtype=np.uint8)
        upperLimit = np.array([upper_h, 255, 255], dtype=np.uint8)

        return lowerLimit, upperLimit


def get_rainbow_color_ranges():
        """Return a dict of color name -> list of (lower, upper) HSV ranges (np.uint8).

        Each color maps to one or more HSV ranges (red needs two ranges due to hue wrap).
        These ranges are tuned for common RGB->HSV mapping used by OpenCV (H:0-179).
        Tune the S and V minimums if you see false positives or missed detections.
        """
        ranges = {
                # Red is split into two ranges (low and high hue) because hue wraps at 179->0
                "red": [
                        (np.array([0, 150, 120], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)),
                        (np.array([170, 150, 120], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8)),
                ],
                # Orange
                "orange": [
                        (np.array([11, 150, 120], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8)),
                ],
                # Yellow
                "yellow": [
                        (np.array([26, 150, 150], dtype=np.uint8), np.array([35, 255, 255], dtype=np.uint8)),
                ],
                # Green
                "green": [
                        (np.array([36, 100, 60], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)),
                ],
                # Blue
                "blue": [
                        (np.array([90, 100, 60], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)),
                ],
                # Purple / Magenta
                "purple": [
                        (np.array([131, 80, 80], dtype=np.uint8), np.array([160, 255, 255], dtype=np.uint8)),
                ],
        }

        return ranges