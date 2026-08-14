# ─────────────────────────────────────────────
#  src/static/main.py  –  live webcam detector
#
#  Usage (from project root):  python -m src.static.main
# ─────────────────────────────────────────────
import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from src.shared.preprocessing import preprocess_frame
from src.shared.smoothing     import LandmarkSmoother, PredictionSmoother
from src.shared.features      import FeatureExtractor
from src.shared.utils         import draw_landmarks, draw_prediction, draw_status


class _LMWrapper:
    def __init__(self, lms):
        self.landmark = lms


def run_detector():
    print("\n" + "="*55)
    print("  Live webcam detector  (Q to quit)")
    print("="*55)

    landmark_smoother   = LandmarkSmoother(window_size=config.LANDMARK_WINDOW_SIZE)
    prediction_smoother = PredictionSmoother(window_size=config.PREDICTION_WINDOW_SIZE)
    feature_extractor   = FeatureExtractor()

    classifier = None
    if os.path.exists(config.MODEL_PATH):
        from src.static.classifier import SignClassifier
        classifier = SignClassifier(model_path=config.MODEL_PATH)
        print("[detector] Classifier loaded.")
    else:
        print(f"[detector] No model found at '{config.MODEL_PATH}' — landmark-only mode.")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL),
        num_hands=config.MAX_NUM_HANDS,
        min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    cap = cv2.VideoCapture(0)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_frame(frame)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]

                if config.SHOW_LANDMARKS:
                    draw_landmarks(frame, _LMWrapper(hand_landmarks))

                raw      = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
                smoothed = landmark_smoother.smooth(raw)
                features = feature_extractor.extract(smoothed)

                if classifier is not None:
                    label, confidence = classifier.predict(features)
                    stable_label = prediction_smoother.update(label)
                    draw_prediction(frame, stable_label, confidence)
                else:
                    draw_status(frame, "Landmark mode — no model loaded", color=(0, 200, 255))

            else:
                landmark_smoother.reset()
                feature_extractor.reset()
                prediction_smoother.reset()
                draw_status(frame, "No hand detected", color=(0, 100, 255))

            draw_status(frame, "Press Q to quit")
            cv2.imshow("Sign Language Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector()
