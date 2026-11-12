import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Map keyboard keys to gesture labels: extend this as you add more letters

KEY_TO_LABEL = {
    ord('a'): 'A', ord('b'): 'B', ord('c'): 'C', ord('d'): 'D',
    ord('e'): 'E', ord('f'): 'F', ord('g'): 'G', ord('h'): 'H',
    ord('i'): 'I', ord('j'): 'J', ord('k'): 'K', ord('l'): 'L',
    ord('m'): 'M', ord('n'): 'N', ord('o'): 'O', ord('p'): 'P',
    ord('q'): 'Q', ord('r'): 'R', ord('s'): 'S', ord('t'): 'T',
    ord('u'): 'U', ord('v'): 'V', ord('w'): 'W', ord('x'): 'X',
    ord('y'): 'Y', ord('z'): 'Z',
}

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures.csv')

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
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['label'] + [f'f{i}' for i in range(126)]  # 21 landmarks * 3
            writer.writerow(header)

        print("Press a-b-l-v-y to save a sample for that letter. Press 'q' to quit.")
        counts = {v: 0 for v in KEY_TO_LABEL.values()}

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # show running counts on screen
            y0 = 30
            for label, c in counts.items():
                cv2.putText(frame, f"{label}: {c}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2)
                y0 += 25

            cv2.imshow("Data Collection - Vision2Braille", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key in KEY_TO_LABEL and results.multi_hand_landmarks:
                label = KEY_TO_LABEL[key]
                # use the first detected hand for now
                features = extract_features(results.multi_hand_landmarks, results.multi_handedness)
                writer.writerow([label] + features)
                counts[label] += 1
                print(f"Saved sample for '{label}' (total: {counts[label]})")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
