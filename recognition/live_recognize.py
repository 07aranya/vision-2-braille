import cv2
import mediapipe as mp
import pickle
import os
from collections import deque, Counter

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'classifier.pkl')

with open(MODEL_PATH, 'rb') as f:
    clf = pickle.load(f)

def extract_features(multi_hand_landmarks, multi_handedness):
    """
    Combines up to 2 hands into a fixed 126-length feature vector
    (42 landmarks x 3 coords). Left hand always occupies the first
    63 values, right hand the next 63. Missing hand = zeros.
    """
    def normalize(hand_landmarks):
        coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        wrist = coords[0]
        scale = ((coords[9][0]-wrist[0])**2 + (coords[9][1]-wrist[1])**2) ** 0.5
        scale = scale if scale > 1e-6 else 1e-6
        feats = []
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

def main():
    cap = cv2.VideoCapture(0)
    STABILITY_WINDOW = 15   # require consistent prediction across this many frames
    recent_preds = deque(maxlen=STABILITY_WINDOW)
    confirmed_letter = None

    print("Starting live recognition. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_pred = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            features = extract_features(results.multi_hand_landmarks, results.multi_handedness)
            current_pred = clf.predict([features])[0]

        recent_preds.append(current_pred)

        # only accept if the same prediction dominates the recent window
        if len(recent_preds) == STABILITY_WINDOW:
            most_common, count = Counter(recent_preds).most_common(1)[0]
            if most_common is not None and count >= STABILITY_WINDOW * 0.8:
                if most_common != confirmed_letter:
                    confirmed_letter = most_common
                    print(f"Recognized: {confirmed_letter}")

        display_text = confirmed_letter if confirmed_letter else "..."
        cv2.putText(frame, f"Letter: {display_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Vision2Braille - Live Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
