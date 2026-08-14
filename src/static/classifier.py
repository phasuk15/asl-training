
#  model loading + inference

import os
import pickle
import numpy as np
import config


class SignClassifier:
    """
    Thin wrapper around a trained sklearn classifier.
    Handles loading, prediction, and confidence reporting.
    """
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        print(f"[SignClassifier] Loaded model from {model_path}")

        # Load label encoder so integer predictions are decoded to letters
        le_path = os.path.join(config.MODELS_DIR, "label_encoder.pkl")
        if os.path.exists(le_path):
            with open(le_path, "rb") as f:
                self.label_encoder = pickle.load(f)
            print(f"[SignClassifier] Label encoder loaded from {le_path}")
        else:
            self.label_encoder = None
            print("[SignClassifier] WARNING: label_encoder.pkl not found — predictions will show as numbers.")

    def predict(self, features: np.ndarray):
        """
        Args:
            features: 1D feature vector
        Returns:
            (label: str, confidence: float)
        """
        features   = features.reshape(1, -1)
        pred_int   = self.model.predict(features)[0]
        confidence = self.model.predict_proba(features).max()

        if self.label_encoder is not None:
            label = self.label_encoder.inverse_transform([pred_int])[0]
        else:
            label = str(pred_int)

        return label, confidence
