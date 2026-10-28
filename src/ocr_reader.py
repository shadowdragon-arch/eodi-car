"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
High-Precision OCR Engine using RapidOCR + Malaysian JPJ Syntax Post-Correction.
"""

import re
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

# Standard & Extended Malaysian Plate Regex:
# Supports: WXY1234, WA1234A, KV1234B, QAA1234C, SAB1234P, F1234, V1234, PUTRAJAYA1234, G1M1234
PLATE_PATTERN = re.compile(
    r"^(?:[A-Z]{1,3}|PUTRAJAYA|PROTON|PERODUA|BAMBEE|RIMAU|SUKOM|PATRIOT|PERFECT|G1M|IM4U|U|V|F|W|B|J|P|K|A|C|D|M|N|R|T|Q[A-Z]{1,2}|S[A-Z]{1,2}|KV)"
    r"[1-9][0-9]{0,3}[A-Z]?$"
)
GENERIC_PLATE_PATTERN = re.compile(r"^[A-Z]{1,3}[1-9][0-9]{0,3}[A-Z]?$")

# Position-specific optical confusion maps
LETTER_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "L": "1", "T": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}
DIGIT_TO_LETTER = {
    "0": "W",  # Or D/Q; handled contextually
    "1": "J",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

_engine = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        _engine = RapidOCR()
    return _engine


def normalise(text: str) -> str:
    """Strips all non-alphanumeric characters and uppercases the string."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def smart_malaysian_plate_clean(raw_text: str) -> str:
    """
    Corrects common OCR character confusions (e.g., O<->0, I<->1, B<->8, S<->5, Z<->2)
    by enforcing Malaysian JPJ structural rules:
    [Prefix Letters (1-3)] + [Digits (1-4, first digit 1-9)] + [Optional Suffix Letter (0-1)]
    """
    s = normalise(raw_text)
    if len(s) < 2 or len(s) > 9:
        return s

    # Special named prefixes
    for special in ("PUTRAJAYA", "PATRIOT", "PERFECT", "BAMBEE", "RIMAU", "SUKOM", "G1M", "IM4U"):
        if s.startswith(special):
            rest = s[len(special):]
            corrected_rest = "".join(LETTER_TO_DIGIT.get(c, c) for c in rest)
            return special + corrected_rest

    # Already strictly valid and free of forbidden JPJ letters (I, O) in prefix/suffix
    if GENERIC_PLATE_PATTERN.match(s) and "I" not in s and "O" not in s:
        return s

    best_candidate = s
    best_cost = 999

    # Test all plausible splits: prefix_len (1..3), digit_len (1..4), suffix_len (0..1)
    n = len(s)
    for p_len in (1, 2, 3):
        for s_len in (0, 1):
            d_len = n - p_len - s_len
            if not (1 <= d_len <= 4):
                continue

            prefix_part = s[:p_len]
            digit_part = s[p_len:p_len + d_len]
            suffix_part = s[p_len + d_len:]

            cost = 0
            new_prefix = []
            for ch in prefix_part:
                if ch.isalpha():
                    # Standard Malaysian plates omit I and O in letters
                    if ch in ("I", "O"):
                        cost += 2
                    new_prefix.append(ch)
                elif ch in DIGIT_TO_LETTER:
                    new_prefix.append(DIGIT_TO_LETTER[ch])
                    cost += 1
                else:
                    cost += 10
                    new_prefix.append(ch)

            new_digits = []
            for idx, ch in enumerate(digit_part):
                if ch.isdigit():
                    if idx == 0 and ch == "0":
                        cost += 10  # Leading zero is invalid on Malaysian plates
                    new_digits.append(ch)
                elif ch in LETTER_TO_DIGIT:
                    mapped = LETTER_TO_DIGIT[ch]
                    if idx == 0 and mapped == "0":
                        cost += 10
                    else:
                        cost += 1
                    new_digits.append(mapped)
                else:
                    cost += 10
                    new_digits.append(ch)

            new_suffix = []
            for ch in suffix_part:
                if ch.isalpha():
                    if ch in ("I", "O", "Z"):
                        cost += 2
                    new_suffix.append(ch)
                elif ch in DIGIT_TO_LETTER:
                    new_suffix.append(DIGIT_TO_LETTER[ch])
                    cost += 1
                else:
                    cost += 10
                    new_suffix.append(ch)

            candidate = "".join(new_prefix) + "".join(new_digits) + "".join(new_suffix)
            if GENERIC_PLATE_PATTERN.match(candidate) and cost < best_cost:
                best_cost = cost
                best_candidate = candidate

    return best_candidate


def is_valid_plate(text: str) -> bool:
    """Checks if cleaned text matches a valid Malaysian vehicle plate format."""
    cleaned = smart_malaysian_plate_clean(text)
    return bool(PLATE_PATTERN.match(cleaned) or GENERIC_PLATE_PATTERN.match(cleaned))


def _sort_boxes_reading_order(result: list) -> list:
    """
    Sorts OCR text boxes in true reading order:
    handles both single-line plates (left-to-right) and two-line square plates (top-to-bottom, then left-to-right).
    """
    if len(result) <= 1:
        return result

    annotated = []
    for item in result:
        box = item[0]
        xs = [pt[0] for pt in box]
        ys = [pt[1] for pt in box]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        h = max(1.0, max(ys) - min(ys))
        annotated.append((cx, cy, h, item))

    avg_h = sum(a[2] for a in annotated) / len(annotated)
    annotated.sort(key=lambda a: a[1])  # Sort by vertical center first

    rows = []
    for a in annotated:
        if not rows or abs(a[1] - rows[-1][0][1]) > avg_h * 0.65:
            rows.append([a])
        else:
            rows[-1].append(a)

    ordered = []
    for row in rows:
        row.sort(key=lambda a: a[0])  # Sort left-to-right within the same line
        ordered.extend(a[3] for a in row)
    return ordered


def preprocess_plate_variants(crop_bgr: np.ndarray) -> list:
    """
    Generates high-contrast image variants for multi-pass OCR:
    1. Upscaled BGR with CLAHE luminance enhancement (best for standard plates)
    2. Inverted CLAHE grayscale (helps when dark-on-light vs light-on-dark contrast varies)
    3. Bilateral filtered + sharpened image (recovers motion-blurred or distant plates)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr.copy()
    h, w = gray.shape[:2]

    # Upscale small crops so characters are at least ~110px tall
    if h < 110:
        scale = 110.0 / max(1, h)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Add a clean border so edge characters aren't clipped by RapidOCR
    gray_padded = cv2.copyMakeBorder(gray, 12, 12, 16, 16, cv2.BORDER_REPLICATE)

    # Variant 1: CLAHE + Bilateral Filter (preserves sharp character edges)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_padded)
    bilateral = cv2.bilateralFilter(enhanced, 5, 45, 45)
    v1 = cv2.cvtColor(bilateral, cv2.COLOR_GRAY2BGR)

    # Variant 2: Inverted polarity (black text on white background)
    inverted = cv2.bitwise_not(bilateral)
    v2 = cv2.cvtColor(inverted, cv2.COLOR_GRAY2BGR)

    # Variant 3: Unsharp mask sharpening on original resized crop
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced, 1.6, gaussian, -0.6, 0)
    v3 = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)

    return [v1, v2, v3]


def ocr_items(image_bgr: np.ndarray) -> list:
    """
    Returns all raw OCR detections in reading order as a list of dicts:
    [{'text': str, 'score': float, 'box': list}, ...]
    Used by check_ocr.py and full-frame fallback detection.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    result, _ = _get_engine()(image_bgr)
    if not result:
        return []
    ordered = _sort_boxes_reading_order(result)
    return [
        {"text": item[1], "score": float(item[2]), "box": item[0]}
        for item in ordered
    ]


def _extract_best_from_result(result: list) -> tuple:
    """Extracts the highest-confidence valid Malaysian plate from a list of RapidOCR boxes."""
    if not result:
        return "", 0.0

    ordered = _sort_boxes_reading_order(result)
    # Filter out tiny low-confidence noise (like dealer phone numbers)
    filtered = [item for item in ordered if float(item[2]) >= 0.35]
    if not filtered:
        filtered = ordered

    candidates = []

    # Candidate A: Each individual text box (in case whole plate is in 1 box)
    for item in filtered:
        raw = normalise(item[1])
        cleaned = smart_malaysian_plate_clean(raw)
        conf = float(item[2])
        candidates.append((cleaned, conf, is_valid_plate(cleaned)))

    # Candidate B: Adjacent pairs (for 2-line plates or split prefix + number boxes)
    for i in range(len(filtered) - 1):
        combined_raw = normalise(filtered[i][1] + filtered[i + 1][1])
        cleaned = smart_malaysian_plate_clean(combined_raw)
        conf = (float(filtered[i][2]) + float(filtered[i + 1][2])) / 2.0
        candidates.append((cleaned, conf, is_valid_plate(cleaned)))

    # Candidate C: All boxes combined
    all_raw = normalise("".join(item[1] for item in filtered))
    all_cleaned = smart_malaysian_plate_clean(all_raw)
    all_conf = sum(float(item[2]) for item in filtered) / len(filtered)
    candidates.append((all_cleaned, all_conf, is_valid_plate(all_cleaned)))

    # Prefer valid plates with highest confidence
    valid_candidates = [c for c in candidates if c[2]]
    if valid_candidates:
        best = max(valid_candidates, key=lambda c: c[1])
        return best[0], best[1]

    # Otherwise return highest-confidence raw cleaned read
    best = max(candidates, key=lambda c: c[1])
    return best[0], best[1]


def ocr_plate_image(crop_bgr: np.ndarray) -> tuple:
    """
    Runs multi-pass OCR across preprocessed variants of the plate crop.
    Returns (plate_text, confidence 0.0-1.0).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    best_text, best_conf, best_valid = "", 0.0, False

    for variant in preprocess_plate_variants(crop_bgr):
        result, _ = _get_engine()(variant)
        if not result:
            continue
        text, conf = _extract_best_from_result(result)
        valid = is_valid_plate(text)

        # Immediately return if we get a high-confidence valid plate (>= 92%)
        if valid and conf >= 0.92:
            return text, conf

        if (valid and not best_valid) or (valid == best_valid and conf > best_conf):
            best_text, best_conf, best_valid = text, conf, valid

    return best_text, best_conf


def read_plate_from_image(cropped_image_path: str) -> tuple:
    """Compatibility wrapper for reading a saved crop from disk."""
    img = cv2.imread(cropped_image_path)
    if img is None:
        return "", 0.0
    return ocr_plate_image(img)