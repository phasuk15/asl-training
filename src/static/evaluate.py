
#  live camera accuracy evaluation

import os
import csv
import time
import random
import argparse
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from src.shared.preprocessing import preprocess_frame
from src.shared.smoothing     import LandmarkSmoother
from src.shared.features      import FeatureExtractor
from src.static.classifier    import SignClassifier
from src.shared.utils         import draw_landmarks

# J and Z are motion-based signs — excluded by default since the
# current model is trained on static images
DEFAULT_SIGNS  = list("ABCDEFGHIKLMNOPQRSTUVWXY")
COUNTDOWN_SECS = 3    # get-ready countdown before recording starts
RESULT_SECS    = 1.5  # how long to show correct / wrong feedback

CORRECT_COLOR = (0,  200,  60)
WRONG_COLOR   = (0,   50, 220)
NEUTRAL_COLOR = (0,  200, 255)

class _LMWrapper:
    def __init__(self, lms): self.landmark = lms


# ── Drawing helpers ────────────────────────────────────────────────────────

def _bar(frame, x, y, w, h, fraction, color, bg=(60, 60, 60)):
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg, -1)
    if fraction > 0:
        cv2.rectangle(frame, (x, y), (x + int(w * fraction), y + h), color, -1)


def _overlay_text(frame, text, pos, scale, color, thickness=2, shadow=True):
    if shadow:
        sx, sy = pos[0] + 2, pos[1] + 2
        cv2.putText(frame, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_ready(frame, letter, countdown_left, idx, total):
    h, w = frame.shape[:2]
    # dark top panel
    cv2.rectangle(frame, (0, 0), (w, 130), (25, 25, 25), -1)
    _overlay_text(frame, f"Sign {idx} of {total}", (16, 30),
                  0.65, (160, 160, 160), 1)
    _overlay_text(frame, "Get ready to sign:", (16, 68),
                  0.9, (210, 210, 210), 1)
    # big letter
    _overlay_text(frame, letter, (w // 2 - 55, h // 2 + 60),
                  5.0, (255, 255, 100), 6)
    # countdown
    secs = int(countdown_left) + 1
    _overlay_text(frame, str(secs), (w - 70, 115),
                  2.2, NEUTRAL_COLOR, 3)
    # bottom bar
    cv2.rectangle(frame, (0, h - 36), (w, h), (25, 25, 25), -1)
    _overlay_text(frame, "S: skip   Q: quit", (10, h - 10),
                  0.5, (140, 140, 140), 1, shadow=False)


def _draw_record(frame, letter, elapsed, record_secs, live_pred, idx, total):
    h, w = frame.shape[:2]
    fraction = min(elapsed / record_secs, 1.0)
    # top panel
    cv2.rectangle(frame, (0, 0), (w, 130), (20, 20, 20), -1)
    _overlay_text(frame, f"Sign {idx} of {total}", (16, 30),
                  0.65, (160, 160, 160), 1)
    # recording indicator
    _overlay_text(frame, "● RECORDING", (16, 70),
                  0.85, (0, 60, 220), 2)
    # target letter (smaller during recording)
    _overlay_text(frame, letter, (w - 110, 115), 2.8, (255, 255, 100), 4)
    # progress bar
    _bar(frame, 16, 90, w - 32, 18, fraction, (0, 180, 220))
    # live prediction hint
    if live_pred:
        _overlay_text(frame, f"Seeing: {live_pred}", (16, h // 2 - 20),
                      1.0, (180, 180, 180), 1)


def _draw_result(frame, letter, predicted, correct):
    h, w = frame.shape[:2]
    color  = CORRECT_COLOR if correct else WRONG_COLOR
    symbol = "CORRECT" if correct else "WRONG"
    alpha  = 0.45
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    _overlay_text(frame, symbol, (w // 2 - 110, h // 2 - 30),
                  2.0, (255, 255, 255), 3)
    if not correct:
        _overlay_text(frame, f"Target: {letter}   Got: {predicted or '—'}",
                      (w // 2 - 160, h // 2 + 40), 0.85, (255, 255, 255), 2)


def _draw_intro(frame, total_signs, model_name):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (40, 60), (w - 40, h - 60), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)

    _overlay_text(frame, "Live Accuracy Evaluation", (70, 120),
                  1.1, (255, 255, 100), 2)
    _overlay_text(frame, f"Model:  {model_name}", (70, 165),
                  0.65, (180, 180, 180), 1, shadow=False)
    _overlay_text(frame, f"Signs:  {total_signs} letters", (70, 195),
                  0.65, (180, 180, 180), 1, shadow=False)
    _overlay_text(frame, f"Countdown: {COUNTDOWN_SECS}s  |  Then hold sign steady", (70, 225),
                  0.62, (180, 180, 180), 1, shadow=False)
    _overlay_text(frame, "Controls:", (70, 270),
                  0.65, (210, 210, 210), 1, shadow=False)
    _overlay_text(frame, "  SPACE  start / next", (70, 298),
                  0.6, (150, 220, 150), 1, shadow=False)
    _overlay_text(frame, "  S      skip current sign", (70, 322),
                  0.6, (150, 220, 150), 1, shadow=False)
    _overlay_text(frame, "  Q      quit and show results", (70, 346),
                  0.6, (150, 220, 150), 1, shadow=False)
    _overlay_text(frame, "Press SPACE to begin", (70, h - 100),
                  1.0, (255, 255, 255), 2)


# ── Results helpers ────────────────────────────────────────────────────────

def _print_summary(results):
    correct = sum(1 for r in results if r["correct"])
    total   = len(results)
    acc     = correct / total if total else 0

    print("\n" + "="*55)
    print(f"  EVALUATION RESULTS  —  {correct}/{total}  ({acc:.0%})")
    print("="*55)
    print(f"  {'Sign':<6} {'Predicted':<12} {'Result'}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*8}")
    for r in results:
        mark = "✓" if r["correct"] else "✗"
        pred = r["predicted"] or "—"
        print(f"  {r['target']:<6} {pred:<12} {mark}")

    print(f"\n  Overall accuracy: {acc:.1%}  ({correct}/{total})")

    # Worst signs
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print("\n  Incorrect predictions:")
        for r in wrong:
            print(f"    {r['target']} → {r['predicted'] or 'no detection'}")

    return acc


def _save_csv(results, acc, path="evaluation_results.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "predicted", "correct", "detection_rate"])
        for r in results:
            writer.writerow([r["target"], r["predicted"] or "",
                             r["correct"], f"{r['detection_rate']:.2f}"])
        writer.writerow([])
        writer.writerow(["overall_accuracy", f"{acc:.4f}"])
    print(f"\n  Results saved → {path}")


# ── Main evaluation loop ───────────────────────────────────────────────────

def run_evaluation(signs, record_secs):
    if not os.path.exists(config.MODEL_PATH):
        print(f"[evaluate] No model found at '{config.MODEL_PATH}'.")
        print("           Run `python -m src.static.train` first.")
        return

    classifier     = SignClassifier(model_path=config.MODEL_PATH)
    lm_smoother    = LandmarkSmoother(window_size=config.LANDMARK_WINDOW_SIZE)
    feat_extractor = FeatureExtractor()

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.HAND_LANDMARKER_MODEL),
        num_hands=1,
        min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )

    results  = []
    sign_idx = 0
    state    = "intro"   # intro → ready → record → result → (next or summary)
    state_t  = time.time()
    predictions_this_round = []
    detection_frames = 0
    total_frames     = 0
    live_pred        = None
    skip_requested   = False

    cap = cv2.VideoCapture(0)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame    = preprocess_frame(frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res      = landmarker.detect(mp_image)
            now      = time.time()
            elapsed  = now - state_t

            hand_visible = bool(res.hand_landmarks)
            live_pred    = None

            if hand_visible:
                lm  = res.hand_landmarks[0]
                if config.SHOW_LANDMARKS:
                    draw_landmarks(frame, _LMWrapper(lm))
                raw      = np.array([[p.x, p.y, p.z] for p in lm])
                smoothed = lm_smoother.smooth(raw)
                features = feat_extractor.extract(smoothed)
                label, conf = classifier.predict(features)
                if conf >= config.MIN_PREDICTION_CONFIDENCE:
                    live_pred = label
                    if state == "record":
                        predictions_this_round.append(label)
                        detection_frames += 1

            if state == "record":
                total_frames += 1

            # ── State machine ──────────────────────────────────────────────
            if state == "intro":
                _draw_intro(frame, len(signs), config.MODEL_PATH)

            elif state == "ready":
                letter = signs[sign_idx]
                countdown_left = COUNTDOWN_SECS - elapsed
                _draw_ready(frame, letter, countdown_left,
                            sign_idx + 1, len(signs))
                if elapsed >= COUNTDOWN_SECS or skip_requested:
                    if skip_requested:
                        # skip: record a miss immediately
                        results.append({
                            "target": letter, "predicted": None,
                            "correct": False, "detection_rate": 0.0,
                        })
                        sign_idx += 1
                        skip_requested = False
                        state = "ready" if sign_idx < len(signs) else "summary"
                    else:
                        predictions_this_round = []
                        detection_frames = 0
                        total_frames     = 0
                        lm_smoother.reset()
                        feat_extractor.reset()
                        state   = "record"
                    state_t = now

            elif state == "record":
                letter = signs[sign_idx]
                _draw_record(frame, letter, elapsed, record_secs,
                             live_pred, sign_idx + 1, len(signs))
                if elapsed >= record_secs or skip_requested:
                    predicted = (Counter(predictions_this_round).most_common(1)[0][0]
                                 if predictions_this_round else None)
                    correct   = (predicted == signs[sign_idx])
                    det_rate  = detection_frames / total_frames if total_frames else 0.0
                    results.append({
                        "target": signs[sign_idx], "predicted": predicted,
                        "correct": correct, "detection_rate": det_rate,
                    })
                    skip_requested = False
                    state   = "result"
                    state_t = now

            elif state == "result":
                r = results[-1]
                _draw_result(frame, r["target"], r["predicted"], r["correct"])
                if elapsed >= RESULT_SECS:
                    sign_idx += 1
                    state    = "ready" if sign_idx < len(signs) else "summary"
                    state_t  = now

            elif state == "summary":
                cv2.destroyAllWindows()
                break

            # ── Key handling ───────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord(" ") and state == "intro":
                state   = "ready"
                state_t = now
            elif key == ord("s") and state in ("ready", "record"):
                skip_requested = True

            cv2.imshow("Live Evaluation — Sign Language", frame)

    cap.release()
    cv2.destroyAllWindows()

    if results:
        acc = _print_summary(results)
        _save_csv(results, acc)
    else:
        print("No results recorded.")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signs", type=str, default=None,
                        help="Letters to test, e.g. ABCDE (default: all except J, Z)")
    parser.add_argument("--no-random", action="store_true",
                        help="Test in alphabetical order instead of random")
    parser.add_argument("--record-secs", type=int, default=3,
                        help="Seconds to hold each sign (default: 3)")
    args = parser.parse_args()

    signs = list(args.signs.upper()) if args.signs else list(DEFAULT_SIGNS)
    if not args.no_random:
        random.shuffle(signs)

    print(f"\n  Signs to test ({len(signs)}): {' '.join(signs)}")
    print(f"  Record window: {args.record_secs}s per sign")

    run_evaluation(signs, args.record_secs)
