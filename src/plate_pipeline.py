"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Two-Stage ANPR Pipeline:
  Stage 1: Fast OpenCV morphological candidate boxes + multi-pass OCR.
  Stage 2: Full-frame RapidOCR deep text detector fallback (guarantees detection on close-ups/angled shots).
"""

import os
import cv2
from detector import find_plate_candidates
from ocr_reader import (
    ocr_plate_image,
    ocr_items,
    is_valid_plate,
    smart_malaysian_plate_clean,
    normalise,
)

MAX_CANDIDATES_TO_OCR = 6


def _fallback_full_frame_search(frame_bgr):
    """
    Uses RapidOCR's deep-learning text detector directly on the frame to locate
    plate text when contour bounding boxes miss (e.g., extreme close-up, glare, or synthetic crops).
    """
    H, W = frame_bgr.shape[:2]
    items = ocr_items(frame_bgr)
    if not items:
        return None

    best = None

    # Check single detected text boxes and adjacent pairs (for 2-line plates)
    groups = [[it] for it in items]
    for i in range(len(items) - 1):
        groups.append([items[i], items[i + 1]])

    for grp in groups:
        raw_text = normalise("".join(g["text"] for g in grp))
        cleaned = smart_malaysian_plate_clean(raw_text)
        if not is_valid_plate(cleaned):
            continue

        avg_conf = sum(g["score"] for g in grp) / len(grp)
        pts = [pt for g in grp for pt in g["box"]]
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]

        x1, y1 = max(0, min(xs)), max(0, min(ys))
        x2, y2 = min(W, max(xs)), min(H, max(ys))
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)

        pad_x, pad_y = int(0.08 * bw), int(0.15 * bh)
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(W, x2 + pad_x), min(H, y2 + pad_y)
        crop = frame_bgr[cy1:cy2, cx1:cx2]

        # Re-verify on the tight crop with CLAHE enhancement to boost confidence
        if crop.size > 0:
            re_text, re_conf = ocr_plate_image(crop)
            if is_valid_plate(re_text) and re_conf > avg_conf:
                cleaned, avg_conf = re_text, re_conf

        if best is None or avg_conf > best["conf"]:
            best = {
                "text": cleaned,
                "conf": avg_conf,
                "box": (cx1, cy1, cx2 - cx1, cy2 - cy1),
                "crop": crop if crop.size > 0 else frame_bgr,
            }

    return best


def read_best_plate(frame_bgr, save_photo_crop: bool = False):
    """
    Returns {'text', 'conf', 'box', 'crop'} for the best VALID Malaysian plate in the frame,
    or None if no valid plate is found.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    H, W = frame_bgr.shape[:2]
    best = None

    # Stage 1: Fast OpenCV contour candidates
    candidates = find_plate_candidates(frame_bgr)[:MAX_CANDIDATES_TO_OCR]
    for (x, y, w, h, _) in candidates:
        pad_y = int(0.10 * h)
        pad_x = int(0.06 * w)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        text, conf = ocr_plate_image(crop)
        if not is_valid_plate(text):
            continue

        if best is None or conf > best["conf"]:
            best = {"text": text, "conf": conf, "box": (x, y, w, h), "crop": crop}
            if conf >= 0.94:
                break

    # Stage 2: Deep-learning full-frame fallback if Stage 1 missed or had < 85% confidence
    if best is None or best["conf"] < 0.85:
        fb = _fallback_full_frame_search(frame_bgr)
        if fb is not None and (best is None or fb["conf"] > best["conf"]):
            best = fb

    if best is not None and save_photo_crop and best["crop"].size > 0:
        crop_path = os.path.join("data", "cropped_plates", "latest_photo_crop.jpg")
        os.makedirs(os.path.dirname(crop_path), exist_ok=True)
        cv2.imwrite(crop_path, best["crop"])

    return best