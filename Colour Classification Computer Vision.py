import cv2
import numpy as np
from util import get_rainbow_color_ranges


# Map color names to BGR drawing colors
DRAW_COLORS = {
    "red": (0, 0, 255),
    "orange": (0, 165, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "purple": (128, 0, 128),
}


def get_mask_for_color(hsv, ranges):
    """Return a cleaned combined mask for the provided hsv and list of (lower, upper) ranges."""
    combined_mask = None
    for (lower, upper) in ranges:
        mask = cv2.inRange(hsv, lower, upper)
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv2.bitwise_or(combined_mask, mask)

    if combined_mask is None:
        return None

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    return combined_mask


def detect_and_draw(frame, hsv, color_name, ranges, min_area=500):
    """Detect regions for a color (one or multiple HSV ranges), draw box and label.

    Returns True if something was drawn.
    """
    combined_mask = get_mask_for_color(hsv, ranges)
    if combined_mask is None:
        return False

    # Find contours and pick the largest meaningful one
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # find largest contour by area
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return False

    x, y, w, h = cv2.boundingRect(largest)
    color_bgr = DRAW_COLORS.get(color_name, (0, 255, 0))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)

    # Draw label with background for readability
    label = color_name
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = max(x, 0)
    text_y = max(y - 8, text_h + 4)
    # background rect
    cv2.rectangle(frame, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y + baseline), color_bgr, -1)
    cv2.putText(frame, label, (text_x + 2, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return True


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions or device availability.")

    color_ranges = get_rainbow_color_ranges()
    DEBUG = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Detect each rainbow color and draw boxes/labels
            combined_debug_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            for name, ranges in color_ranges.items():
                mask = get_mask_for_color(hsv, ranges)
                if mask is not None:
                    # accumulate for debug view
                    combined_debug_mask = cv2.bitwise_or(combined_debug_mask, mask)
                # draw boxes/labels regardless of debug
                detect_and_draw(frame, hsv, name, ranges, min_area=800)

            cv2.imshow('Webcam', frame)

            # Toggle debug masks with 'd'. When enabled, show per-color and combined masks.
            if DEBUG:
                for name, ranges in color_ranges.items():
                    mask = get_mask_for_color(hsv, ranges)
                    if mask is None:
                        continue
                    # Show each mask in a small window named by color
                    cv2.imshow(f"mask_{name}", mask)
                cv2.imshow('combined_masks', combined_debug_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                DEBUG = not DEBUG
                # close all mask windows when turning off debug
                if not DEBUG:
                    for name in color_ranges.keys():
                        cv2.destroyWindow(f"mask_{name}")
                    cv2.destroyWindow('combined_masks')

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()