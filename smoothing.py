import numpy as np
from collections import deque, Counter


class LandmarkSmoother:
    """
    Temporal moving-average smoother over raw landmark arrays
    """
    def __init__(self, window_size: int = 5):
        self.buffer = deque(maxlen=window_size)

    def smooth(self, landmarks: np.ndarray) -> np.ndarray:
        self.buffer.append(landmarks)
        return np.mean(self.buffer, axis=0)

    def reset(self):
        self.buffer.clear()


class PredictionSmoother:
    """
    Majority-vote smoother over recent classifier predictions
    """
    def __init__(self, window_size: int = 10):
        self.buffer = deque(maxlen=window_size)

    def update(self, prediction: str) -> str:
        self.buffer.append(prediction)
        return Counter(self.buffer).most_common(1)[0][0]

    def reset(self):
        self.buffer.clear()