"""
EodiCar ANPR Prototype - Samsan Tech Sdn. Bhd.
Milestone 1: Environment Verification and Basic Image Processing
"""

import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_synthetic_car_image(save_path: str, plate_text: str = "WXY 1234") -> str:
    """
    Creates a synthetic test image with a Malaysian number plate ('WXY 1234')
    so you can test the pipeline immediately without needing a real photo first.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create a blank canvas: Height=480, Width=640, 3 channels (BGR)
    img = np.full((480, 640, 3), 60, dtype=np.uint8)

    # Draw a mock car body and Malaysian black/white number plate
    cv2.rectangle(img, (120, 140), (520, 380), (180, 80, 40), -1)
    cv2.rectangle(img, (215, 268), (425, 328), (12, 12, 12), -1)
    cv2.rectangle(img, (215, 268), (425, 328), (220, 220, 220), 2)
    cv2.putText(
        img,
        plate_text,
        (232, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (248, 248, 248),
        3,
        cv2.LINE_AA,
    )

    cv2.imwrite(save_path, img)
    print(f"[INFO] Created synthetic test image at: {save_path}")
    return save_path


def load_and_inspect_image(image_path: str, show_gui: bool = True):
    """
    Loads an image, validates it, prints its matrix shape, and optionally displays it.
    """
    print("=" * 60)
    print("EodiCar ANPR - Milestone 1: Image Loader & Inspector")
    print("=" * 60)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[ERROR] Image not found at: '{image_path}'")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise ValueError(f"[ERROR] OpenCV could not decode '{image_path}'. File may be corrupted.")

    height, width, channels = image_bgr.shape
    print("[SUCCESS] Image loaded successfully!")
    print(f"  -> File Path     : {os.path.abspath(image_path)}")
    print(f"  -> Dimensions    : {width}px Width x {height}px Height")
    print(f"  -> NumPy Shape   : {image_bgr.shape} (Height, Width, Channels)")
    print("  -> Color Format  : OpenCV BGR")
    print("=" * 60)

    if show_gui:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(7, 5))
        plt.imshow(image_rgb)
        plt.title(f"Milestone 1 Test | Resolution: {width}x{height}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EodiCar Milestone 1 Image Loader")
    parser.add_argument(
        "--image",
        default=os.path.join("data", "test_images", "sample_car.jpg"),
        help="Path to test image",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without opening a Matplotlib window",
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        create_synthetic_car_image(args.image)

    load_and_inspect_image(args.image, show_gui=not args.no_gui)