
import numpy as np

# Joint angles
_JOINT_TRIPLETS = [
    (0,  1,  2), (1,  2,  3), (2,  3,  4),   # thumb
    (0,  5,  6), (5,  6,  7), (6,  7,  8),   # index
    (0,  9, 10), (9, 10, 11), (10, 11, 12),  # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16), # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20), # pinky
]


def normalise_landmarks(landmarks: np.ndarray) -> np.ndarray:
    # Centre on wrist (lm 0) and scale by distance to middle-finger MCP (lm 9)
    landmarks = landmarks - landmarks[0]
    scale = np.linalg.norm(landmarks[9])  # lm 9 is a stable mid-palm reference
    if scale > 0:
        landmarks = landmarks / scale
    return landmarks.flatten()


def compute_angles(landmarks: np.ndarray) -> np.ndarray:
    """Compute 15 joint flexion angles from _JOINT_TRIPLETS."""
    angles = []
    for a, b, c in _JOINT_TRIPLETS:
        v1 = landmarks[a] - landmarks[b]
        v2 = landmarks[c] - landmarks[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-7 or n2 < 1e-7:  # guard against degenerate/zero-length segments
            angles.append(0.0)
        else:
            cos_a = np.dot(v1, v2) / (n1 * n2)
            angles.append(np.arccos(np.clip(cos_a, -1.0, 1.0)))
    return np.array(angles, dtype=np.float32)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """Return 78-dim feature vector (63 normalised coords + 15 joint angles) from a (21, 3) landmark array."""
    # Angles instead of inter-frame velocity keep train (single images) and inference (live video) consistent.
    return np.concatenate([normalise_landmarks(landmarks), compute_angles(landmarks)])


class FeatureExtractor:
    """
    Stateless wrapper around extract_features() for use in the webcam loop.
    Kept as a class so main.py can call .reset() without changes.
    """
    def extract(self, landmarks: np.ndarray) -> np.ndarray:
        return extract_features(landmarks)

    def reset(self):
        pass


# Body-aware feature extraction
# Upper-body pose landmark indices (MediaPipe Pose / Holistic):
#   0=nose, 11=L-shoulder, 12=R-shoulder, 13=L-elbow, 14=R-elbow,
#   23=L-hip, 24=R-hip
_POSE_INDICES = [0, 11, 12, 13, 14, 23, 24]

def normalise_pose(pose_landmarks: np.ndarray) -> np.ndarray:
    """Return 7 upper-body pose landmarks (21-dim) centred on the shoulder midpoint and scaled by shoulder width."""
    pts = pose_landmarks[_POSE_INDICES]           # (7, 3)
    l_shoulder = pose_landmarks[11]
    r_shoulder = pose_landmarks[12]
    origin = (l_shoulder + r_shoulder) / 2.0     # shoulder midpoint
    scale = np.linalg.norm(l_shoulder - r_shoulder)
    if scale < 1e-7:
        scale = 1.0
    return ((pts - origin) / scale).flatten()     # (21,)


def hand_body_relation(hand_wrist_world: np.ndarray,
                       pose_landmarks: np.ndarray) -> np.ndarray:
    """Return 6 features encoding wrist position relative to shoulder midpoint and nose."""
    l_shoulder = pose_landmarks[11]
    r_shoulder = pose_landmarks[12]
    shoulder_mid = (l_shoulder + r_shoulder) / 2.0
    scale = np.linalg.norm(l_shoulder - r_shoulder)
    if scale < 1e-7:
        scale = 1.0
    nose = pose_landmarks[0]
    to_shoulder = (hand_wrist_world - shoulder_mid) / scale
    to_nose = (hand_wrist_world - nose) / scale
    return np.concatenate([to_shoulder, to_nose]).astype(np.float32)


def extract_body_features(hand_landmarks: np.ndarray,
                          pose_landmarks: np.ndarray,
                          dominant_side: str = "right") -> np.ndarray:
    """Return 105-dim vector: 63 hand coords + 15 angles + 21 pose + 6 body-relation."""
    hand_feats = extract_features(hand_landmarks)             # 78
    pose_feats = normalise_pose(pose_landmarks)               # 21
    # Use the pose wrist landmark so both hand and body are in the same
    # coordinate frame (image-normalised pixel coords from Holistic).
    wrist_idx = 16 if dominant_side == "right" else 15
    relation = hand_body_relation(pose_landmarks[wrist_idx], pose_landmarks)  # 6
    return np.concatenate([hand_feats, pose_feats, relation]).astype(np.float32)
