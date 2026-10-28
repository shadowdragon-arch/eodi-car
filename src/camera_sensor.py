"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Live camera sensor: detects plate regions, OCRs them, displays live HUD telemetry,
and locks only after several agreeing reads.
"""

import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2

from plate_pipeline import read_best_plate

OCR_EVERY_N_FRAMES = 3     # run detection+OCR on every 3rd frame
VOTE_WINDOW = 5            # look at the last 5 valid reads
LOCK_VOTES = 3             # the same plate must appear 3 times in that window
MIN_LOCK_CONF = 0.55       # ignore very unsure reads
TIMEOUT_SECONDS = 30       # give up after 30 seconds


def _open_camera():
    """Opens webcam safely across Windows (DirectShow) and Linux/macOS."""
    if os.name == "nt":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(0)


def run_live_camera_sensor(scan_label: str = "AUTO") -> tuple:
    print("\n" + "=" * 60)
    print(f"  [AUTOMATIC SENSOR ENGAGED] Mode: {scan_label}")
    print("  -> Hold the plate in front of the camera, 30-60 cm away, in good light.")
    print(f"  -> Auto-locks after {LOCK_VOTES} matching reads. Timeout: {TIMEOUT_SECONDS}s.")
    print("  -> Press [Q] or [ESC] to abort.")
    print("=" * 60)

    cap = _open_camera()
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return None, None, "", 0.0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window = f"EodiCar Auto Sensor ({scan_label})"
    history = deque(maxlen=VOTE_WINDOW)
    locked = None
    last_candidate = None
    frame_count = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        display = frame.copy()
        elapsed = time.time() - start
        remaining = max(0, int(TIMEOUT_SECONDS - elapsed))

        # Top HUD Bar
        cv2.rectangle(display, (0, 0), (display.shape[1], 48), (20, 20, 20), -1)
        cv2.putText(
            display,
            f"EODICAR {scan_label} | Time Left: {remaining}s | [Q] Cancel",
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )

        if elapsed > TIMEOUT_SECONDS:
            print(f"\n[TIMEOUT] No plate locked within {TIMEOUT_SECONDS} seconds.")
            break

        if frame_count % OCR_EVERY_N_FRAMES == 0:
            best = read_best_plate(frame)
            if best and best["conf"] >= MIN_LOCK_CONF:
                best["frame"] = frame.copy()
                history.append(best)
                last_candidate = best

                winner, votes = Counter(h["text"] for h in history).most_common(1)[0]
                if votes >= LOCK_VOTES:
                    locked = max(
                        (h for h in history if h["text"] == winner),
                        key=lambda h: h["conf"],
                    )
                    x, y, w, h = locked["box"]
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 4)
                    cv2.putText(
                        display,
                        f"LOCKED: {locked['text']} ({locked['conf']*100:.1f}%)",
                        (x, max(65, y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window, display)
                    cv2.waitKey(650)
                    break

        # Draw live candidate tracking feedback between OCR frames
        if last_candidate is not None:
            lx, ly, lw, lh = last_candidate["box"]
            winner, votes = Counter(h["text"] for h in history).most_common(1)[0]
            cv2.rectangle(display, (lx, ly), (lx + lw, ly + lh), (0, 180, 255), 2)
            cv2.putText(
                display,
                f"{last_candidate['text']} ({last_candidate['conf']*100:.0f}%) | Votes: {votes}/{LOCK_VOTES}",
                (lx, max(65, ly - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            print("\n[INFO] Scan aborted by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if locked is None:
        return None, None, "", 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_img_path = os.path.join("data", "test_images", f"live_{scan_label.lower()}_{ts}.jpg")
    crop_path = os.path.join("data", "cropped_plates", f"live_crop_{ts}.jpg")
    os.makedirs(os.path.dirname(saved_img_path), exist_ok=True)
    os.makedirs(os.path.dirname(crop_path), exist_ok=True)
    cv2.imwrite(saved_img_path, locked["frame"])
    cv2.imwrite(crop_path, locked["crop"])
    print(f"\n[AUTO-CAPTURE SUCCESS] Photo saved to: {saved_img_path}")
    print(f"[LOCK] Plate '{locked['text']}' ({locked['conf']*100:.1f}%) confirmed by {LOCK_VOTES}+ reads.")

    return saved_img_path, crop_path, locked["text"], locked["conf"]