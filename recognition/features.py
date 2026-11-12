from typing import List, Optional, Tuple

import numpy as np

# Grade 1 Braille dot patterns (1..6) for A-Z and space
BRAILLE_PATTERNS = {
    'A': [1], 'B': [1, 2], 'C': [1, 4], 'D': [1, 4, 5], 'E': [1, 5],
    'F': [1, 2, 4], 'G': [1, 2, 4, 5], 'H': [1, 2, 5], 'I': [2, 4], 'J': [2, 4, 5],
    'K': [1, 3], 'L': [1, 2, 3], 'M': [1, 3, 4], 'N': [1, 3, 4, 5], 'O': [1, 3, 5],
    'P': [1, 2, 3, 4], 'Q': [1, 2, 3, 4, 5], 'R': [1, 2, 3, 5], 'S': [2, 3, 4], 'T': [2, 3, 4, 5],
    'U': [1, 3, 6], 'V': [1, 2, 3, 6], 'W': [2, 4, 5, 6], 'X': [1, 3, 4, 6], 'Y': [1, 3, 4, 5, 6], 'Z': [1, 3, 5, 6],
    ' ': [],
}

DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_MAX_DISTANCE = 2.2


def extract_features(multi_hand_landmarks, multi_handedness) -> List[float]:
    """
    Combine up to 2 hands into a fixed 126-length feature vector
    (42 landmarks x 3 coords). Left hand always occupies the first
    63 values, right hand the next 63. Missing hand = zeros.
    """
    def normalize(hand_landmarks) -> List[float]:
        coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        wrist = coords[0]
        scale = ((coords[9][0] - wrist[0]) ** 2 + (coords[9][1] - wrist[1]) ** 2) ** 0.5
        scale = scale if scale > 1e-6 else 1e-6
        feats: List[float] = []
        for (x, y, z) in coords:
            feats.extend([(x - wrist[0]) / scale, (y - wrist[1]) / scale, z / scale])
        return feats

    left_feats = [0.0] * 63
    right_feats = [0.0] * 63

    for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        label = handedness.classification[0].label  # 'Left' or 'Right'
        feats = normalize(hand_landmarks)
        if label == 'Left':
            left_feats = feats
        else:
            right_feats = feats

    return left_feats + right_feats


def predict_gesture(
    clf,
    features: List[float],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> Tuple[Optional[str], float, float]:
    """
    Predict a gesture label with confidence and nearest-neighbor distance gating.
    Returns (label_or_none, confidence, mean_neighbor_distance).
    """
    sample = [features]
    distances, _ = clf.kneighbors(sample, n_neighbors=clf.n_neighbors)
    mean_dist = float(np.mean(distances[0]))

    probs = clf.predict_proba(sample)[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    label = str(clf.classes_[best_idx])

    if confidence < min_confidence or mean_dist > max_distance:
        return None, confidence, mean_dist

    return label, confidence, mean_dist


def draw_braille_cell(
    frame,
    letter: Optional[str],
    x: int = 20,
    y: int = 135,
    title: str = "BRAILLE",
) -> None:
    """Draw a 2x3 Braille cell overlay on an OpenCV BGR frame."""
    import cv2

    active_dots = set(BRAILLE_PATTERNS.get(letter, [])) if letter else set()
    box_w, box_h = 84, 118

    # Background card
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (28, 28, 32), -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (90, 90, 100), 1)
    cv2.putText(frame, f"{title}: {letter or '-'}", (x + 6, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

    # Standard 2x3 Braille coordinates: dots 1,2,3 left; 4,5,6 right
    dot_positions = {
        1: (x + 26, y + 40),
        2: (x + 26, y + 68),
        3: (x + 26, y + 96),
        4: (x + 58, y + 40),
        5: (x + 58, y + 68),
        6: (x + 58, y + 96),
    }

    for dot_id, center in dot_positions.items():
        if dot_id in active_dots:
            cv2.circle(frame, center, 9, (0, 220, 120), -1)
            cv2.circle(frame, center, 9, (255, 255, 255), 1)
        else:
            cv2.circle(frame, center, 8, (75, 75, 85), 2)

