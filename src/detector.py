"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
High-Recall Multi-Stage Plate Region Candidate Finder (OpenCV).
Supports standard rectangular plates, 2-line square plates, and close-up webcam scans.
"""

import os
import cv2
import numpy as np

WORK_WIDTH = 640


def find_plate_candidates(image_bgr: np.ndarray, work_width: int = WORK_WIDTH) -> list:
    """
    Returns a deduplicated list of plate-like bounding boxes (x, y, w, h, score)
    in ORIGINAL image coordinates, sorted best-first.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []

    h0, w0 = image_bgr.shape[:2]
    scale = work_width / float(w0) if w0 > work_width else 1.0
    small = (
        cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale != 1.0
        else image_bgr
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blurred = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    # 1. Vertical Sobel Edges (captures dense vertical strokes of characters)
    sobel = cv2.convertScaleAbs(cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3))
    _, th_sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. TopHat & BlackHat Morphology (captures bright text on dark plate AND dark text on white plate)
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    tophat = cv2.morphologyEx(gray_eq, cv2.MORPH_TOPHAT, rect_kern)
    blackhat = cv2.morphologyEx(gray_eq, cv2.MORPH_BLACKHAT, rect_kern)
    _, th_top = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_black = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Generate binary maps with multiple closing kernels (for distant, close-up, and 2-line plates)
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (23, 5)),
        cv2.getStructuringElement(cv2.MORPH_RECT, (13, 7)),  # 2-line square plates
    ]

    H, W = small.shape[:2]
    raw_candidates = []

    for base_bin in (th_sobel, th_top, th_black):
        for kern in kernels:
            closed = cv2.morphologyEx(base_bin, cv2.MORPH_CLOSE, kern)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if h < 12 or w < 30:
                    continue
                aspect = w / float(h)
                area_ratio = (w * h) / float(W * H)
                extent = cv2.contourArea(c) / float(w * h)

                # Supports 2-line square plates (1.2+) up to wide plates (6.5)
                if not (1.2 <= aspect <= 6.5):
                    continue
                # Supports distant plates (0.3% area) up to close-up webcam scans (85% area)
                if not (0.003 <= area_ratio <= 0.85):
                    continue
                if extent < 0.28:
                    continue

                # Measure internal edge density to prefer true character blocks over flat bumpers
                roi_sobel = th_sobel[y:y + h, x:x + w]
                edge_density = float(np.count_nonzero(roi_sobel)) / float(w * h)
                if edge_density < 0.08:
                    continue

                # Ideal plate aspect ratio bonus (~3.2 for standard, ~1.6 for square)
                aspect_bonus = 1.2 if (2.2 <= aspect <= 5.2 or 1.3 <= aspect <= 1.9) else 0.85
                score = extent * edge_density * aspect_bonus * (min(area_ratio, 0.25) ** 0.5)

                orig_box = (
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale),
                    float(score),
                )
                raw_candidates.append(orig_box)

    raw_candidates.sort(key=lambda t: t[4], reverse=True)

    # Non-Maximum Suppression (IoU deduplication) so we don't OCR duplicate overlapping boxes
    deduped = []
    for cand in raw_candidates:
        cx, cy, cw, ch, _ = cand
        overlap = False
        for ex, ey, ew, eh, _ in deduped:
            ix1, iy1 = max(cx, ex), max(cy, ey)
            ix2, iy2 = min(cx + cw, ex + ew), min(cy + ch, ey + eh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (cw * ch) + (ew * eh) - inter
            if union > 0 and (inter / float(union)) > 0.45:
                overlap = True
                break
        if not overlap:
            deduped.append(cand)

    return deduped


def detect_and_crop_plate(image_path: str, save_crop: bool = True, show_plot: bool = False):
    """
    Compatibility wrapper for photo mode.
    Returns (crop, box) or (None, None) if no plate-like region is found.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found at: '{image_path}'")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"[ERROR] Could not decode image at: '{image_path}'")

    candidates = find_plate_candidates(image)
    if not candidates:
        return None, None

    x, y, w, h, _ = candidates[0]
    pad_y = int(0.10 * h)
    pad_x = int(0.06 * w)
    H, W = image.shape[:2]
    crop = image[max(0, y - pad_y):min(H, y + h + pad_y), max(0, x - pad_x):min(W, x + w + pad_x)]

    if save_crop and crop.size > 0:
        crop_path = os.path.join("data", "cropped_plates", "sample_plate_crop.jpg")
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
        cv2.imwrite(crop_path, crop)
    return crop, (x, y, w, h)