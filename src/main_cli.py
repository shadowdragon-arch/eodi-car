"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Milestone 7 & 8: Complete Interactive Demonstration Terminal
Supports BOTH Live Camera Auto-Sensor AND Existing Photo Testing!
"""

import os
import sys
import cv2

from plate_pipeline import read_best_plate
from camera_sensor import run_live_camera_sensor
from milestone1_image_loader import create_synthetic_car_image
from database import (
    init_db,
    is_vehicle_currently_parked,
    verify_plate_text,
    handle_entry,
    handle_exit,
    list_all_sessions,
)


def process_live_camera_scan(mode: str):
    """
    Triggers the live webcam sensor:
    - mode = 'ENTRY': scans plate via camera, saves photo to data/test_images/, logs entry.
    - mode = 'EXIT' : scans plate via camera, saves photo to data/test_images/, logs exit & fee.
    - mode = 'SMART': scans plate via camera, checks DB automatically -> Entry if new, Exit if already parked!
    """
    saved_img_path, crop_path, raw_text, confidence = run_live_camera_sensor(scan_label=mode)

    if not saved_img_path:
        print("[INFO] No plate was locked. Returning to menu.")
        return

    final_plate = verify_plate_text(raw_text, confidence, auto_mode=True)
    if not final_plate:
        print("[INFO] Scan cancelled by operator.")
        return

    if mode == "SMART":
        if is_vehicle_currently_parked(final_plate):
            print(f"\n[SMART SENSOR] Vehicle '{final_plate}' has an active session -> Routing to EXIT!")
            handle_exit(final_plate)
        else:
            print(f"\n[SMART SENSOR] Vehicle '{final_plate}' is not inside yet -> Routing to ENTRY!")
            handle_entry(final_plate, confidence, saved_img_path)
    elif mode == "ENTRY":
        handle_entry(final_plate, confidence, saved_img_path)
    else:
        handle_exit(final_plate)


def process_existing_photo_scan(is_entry: bool):
    """Runs the ANPR pipeline on an existing image file inside data/test_images/."""
    print("\n" + "-" * 60)
    test_dir = os.path.join("data", "test_images")
    os.makedirs(test_dir, exist_ok=True)

    # Ensure sample_car.jpg exists so default press-Enter works out of the box
    sample_default = os.path.join(test_dir, "sample_car.jpg")
    if not os.path.exists(sample_default):
        create_synthetic_car_image(sample_default)

    available = sorted(
        f for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Available photos in data/test_images/: {', '.join(available) or 'none'}")

    img_name = input("Enter image filename (default: sample_car.jpg): ").strip()
    if img_name == "":
        img_name = "sample_car.jpg"

    image_path = os.path.join(test_dir, img_name)

    if not os.path.exists(image_path):
        print(f"[ERROR] Could not find '{image_path}'. Please check the spelling.")
        return

    print("\n" + "*" * 60)
    print(f"--- PROCESSING PHOTO {'ENTRY' if is_entry else 'EXIT'} SCAN: {img_name} ---")
    print("*" * 60)

    image = cv2.imread(image_path)
    if image is None:
        print("[ERROR] OpenCV could not read this image. The file may be corrupted.")
        return

    best = read_best_plate(image, save_photo_crop=True)
    if best is None:
        print("[ERROR] No valid Malaysian plate was found in this image.")
        print("        Try a closer, sharper photo with the plate facing the camera.")
        return

    plate_text, confidence = best["text"], best["conf"]
    print(f"[OCR] Read '{plate_text}' with confidence {confidence * 100:.1f}%")

    final_plate = verify_plate_text(plate_text, confidence, auto_mode=True)
    if not final_plate:
        print("[INFO] Scan cancelled by operator.")
        return

    if is_entry:
        handle_entry(final_plate, confidence, image_path)
    else:
        handle_exit(final_plate)


def main():
    init_db()

    while True:
        print("\n" + "=" * 65)
        print("   EODICAR ANPR PARKING SYSTEM - IOI CITY MALL PROTOTYPE")
        print("=" * 65)
        print("  --- LIVE WEBCAM SENSOR MODE (Auto-Detects & Saves Photo) ---")
        print("  [1] Start Camera Sensor -> Vehicle ENTRY")
        print("  [2] Start Camera Sensor -> Vehicle EXIT")
        print("  [3] Start Camera Sensor -> SMART AUTO-GATE (Auto Entry/Exit!)")
        print("\n  --- EXISTING PHOTO MODE (From data/test_images/) ---")
        print("  [4] Scan Existing Photo -> Vehicle ENTRY")
        print("  [5] Scan Existing Photo -> Vehicle EXIT")
        print("\n  --- DATABASE & SYSTEM ---")
        print("  [6] View Parking Lot Database & Live Fee Timer")
        print("  [7] Quit System")

        choice = input("\nSelect an option (1-7): ").strip()

        if choice == "1":
            process_live_camera_scan(mode="ENTRY")
        elif choice == "2":
            process_live_camera_scan(mode="EXIT")
        elif choice == "3":
            process_live_camera_scan(mode="SMART")
        elif choice == "4":
            process_existing_photo_scan(is_entry=True)
        elif choice == "5":
            process_existing_photo_scan(is_entry=False)
        elif choice == "6":
            list_all_sessions()
        elif choice == "7":
            print("Shutting down EodiCar System... Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please type a number from 1 to 7.")


if __name__ == "__main__":
    main()