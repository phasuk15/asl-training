
import cv2
import config

# Hand connection pairs (MediaPipe hand topology)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),         # thumb
    (0,5),(5,6),(6,7),(7,8),         # index
    (0,9),(9,10),(10,11),(11,12),    # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17),            # palm
]


def draw_landmarks(frame, hand_landmarks):
    """
    Manually draw hand skeleton using landmark x/y pixel coordinates
    """
    h, w, _ = frame.shape
    points = []

    for lm in hand_landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 1)


def draw_prediction(frame, label: str, confidence: float):
    """Overlay predicted sign and confidence on the frame."""
    text = f"{label}  ({confidence:.0%})"
    cv2.putText(
        frame, text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        config.FONT_COLOR,
        config.FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_status(frame, message: str, color=(0, 200, 255)):
    """Overlay a small status message at the bottom of the frame."""
    h, _, _ = frame.shape
    cv2.putText(
        frame, message,
        (10, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, color, 1, cv2.LINE_AA,
    )