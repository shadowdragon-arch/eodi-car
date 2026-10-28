"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Diagnostic utility: Shows raw RapidOCR detections AND final ANPR pipeline output
for every image in data/test_images/.
"""

import os
import cv2
from ocr_reader import ocr_items
from plate_pipeline import read_best_plate
from milestone1_image_loader import create_synthetic_car_image

folder = os.path.join("data", "test_images")
os.makedirs(folder, exist_ok=True)

# Ensure at least sample_car.jpg exists for immediate testing
sample_path = os.path.join(folder, "sample_car.jpg")
if not os.path.exists(sample_path):
    create_synthetic_car_image(sample_path)

image_files = sorted(
    f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
)

if not image_files:
    print(f"[INFO] No images found in '{folder}'.")
else:
    for name in image_files:
        path = os.path.join(folder, name)
        img = cv2.imread(path)
        if img is None:
            print(f"\n=== {name}: could not read ===")
            continue

        print(f"\n=== {name} ({img.shape[1]}x{img.shape[0]}) ===")
        items = ocr_items(img)
        if not items:
            print("  Raw OCR: (no text detected)")
        for it in items:
            print(f"  Raw OCR: '{it['text']}'  score={it['score'] * 100:.1f}%")

        best = read_best_plate(img)
        if best:
            print(f"  -> [FINAL ANPR LOCK]: '{best['text']}' | Confidence: {best['conf'] * 100:.1f}% | Box: {best['box']}")
        else:
            print("  -> [FINAL ANPR LOCK]: None (No valid Malaysian plate matched)")