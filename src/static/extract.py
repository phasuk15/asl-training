# ─────────────────────────────────────────────
#  src/static/extract.py  –  landmark extraction from dataset images
#
#  Usage (from project root):  python -m src.static.extract
#          (reads data/raw/asl_alphabet_train/, writes data/processed/hand_data.csv)
# ─────────────────────────────────────────────
import os
import cv2
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from src.shared.features import extract_features

# Number of augmented variants to write per detected hand.
# Each variant applies noise + scale jitter + small rotation.
# Set to 0 to disable augmentation.
_AUG_PER_SAMPLE = 3


def _preprocess_dataset_image(img):
    """Resize large images and apply CLAHE contrast enhancement.
    No horizontal flip — dataset images are not mirror-view like webcam frames.
    """
    h, w = img.shape[:2]
    if max(h, w) > 640:
        scale = 640 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    img = cv2.merge((l, a, b))
    return cv2.cvtColor(img, cv2.COLOR_LAB2BGR)


def _augment_raw_landmarks(raw: np.ndarray, rng: np.random.Generator) -> list:
    """
    Generate _AUG_PER_SAMPLE augmented variants of a (21, 3) raw landmark array.
    Augments on raw landmarks (before normalisation) to preserve the effect
    through the normalisation pipeline.

    Augmentations applied simultaneously per variant:
        - Gaussian noise   : simulates MediaPipe detection jitter
        - Scale jitter     : simulates variation in hand size / distance to camera
        - 2-D rotation     : simulates wrist roll (rotation in the image plane)
    """
    variants = []
    for _ in range(_AUG_PER_SAMPLE):
        lm = raw.copy()

        # Gaussian noise — std ≈ 0.5% of image width in normalised coords
        lm += rng.normal(0.0, 0.005, lm.shape)

        # Scale jitter ±8%
        lm *= rng.uniform(0.92, 1.08)

        # In-plane rotation ±12°
        theta = rng.uniform(-0.21, 0.21)
        c, s = np.cos(theta), np.sin(theta)
        x, y = lm[:, 0].copy(), lm[:, 1].copy()
        lm[:, 0] = c * x - s * y
        lm[:, 1] = s * x + c * y

        variants.append(lm)
    return variants


def extract_landmarks(
    dataset_dir=config.DATASET_DIR,
    output_csv=config.DATA_CSV,
    images_per_class=config.IMAGES_PER_CLASS,
):
    print("\n" + "="*55)
    print("  Extracting landmarks from dataset")
    print("="*55)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    class_dirs = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    print(f"  Found {len(class_dirs)} classes: {class_dirs}")
    print(f"  Augmentation: {_AUG_PER_SAMPLE} variants per sample ({_AUG_PER_SAMPLE + 1}× dataset)\n")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,  # low threshold catches tricky signs like T, N, M
        min_hand_presence_confidence=0.3,
    )

    rng = np.random.default_rng(42)  # fixed seed → reproducible augmentation
    total_saved = 0
    total_skip  = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)

            for label in class_dirs:
                label_dir = os.path.join(dataset_dir, label)
                images = [
                    os.path.join(label_dir, fn)
                    for fn in sorted(os.listdir(label_dir))
                    if os.path.splitext(fn)[1].lower() in config.SUPPORTED_EXTS
                ]
                if images_per_class:
                    images = images[:images_per_class]

                saved = skip = 0
                for img_path in images:
                    img = cv2.imread(img_path)
                    if img is None:
                        skip += 1
                        continue

                    img = _preprocess_dataset_image(img)
                    mp_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    )
                    try:
                        res = landmarker.detect(mp_img)
                    except Exception:
                        skip += 1
                        continue

                    if not res.hand_landmarks:
                        skip += 1
                        continue

                    raw = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]])

                    # Write original + augmented variants
                    for variant in [raw] + _augment_raw_landmarks(raw, rng):
                        writer.writerow([label.upper()] + extract_features(variant).tolist())
                    saved += 1

                total_saved += saved
                total_skip  += skip
                rows_written = saved * (_AUG_PER_SAMPLE + 1)
                print(f"  [{label.upper():>2}]  detected: {saved:4d}  skipped: {skip:4d}"
                        f"  rows written: {rows_written:5d}")

    total_rows = total_saved * (_AUG_PER_SAMPLE + 1)
    print(f"\n  ✅  Extraction complete — {total_rows} rows → {output_csv}")
    print(f"      ({total_saved} original + {total_saved * _AUG_PER_SAMPLE} augmented)")
    print(f"      Skipped (no hand detected): {total_skip}")


if __name__ == "__main__":
    extract_landmarks()
