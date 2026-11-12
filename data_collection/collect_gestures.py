import csv
import os
import sys

import cv2
import mediapipe as mp

# Add parent directory to path to import recognition modules if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recognition')))
from features import extract_features  # Shared feature extractor

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# All 26 uppercase letters A-Z
KEY_TO_LABEL = {ord(c.lower()): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
KEY_TO_LABEL.update({ord(c.upper()): c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures.csv')


def load_existing_counts(csv_path):
    counts = {chr(c): 0 for c in range(ord('A'), ord('Z') + 1)}
    if os.path.isfile(csv_path):
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if row and row[0] in counts:
                    counts[row[0]] += 1
    return counts


def draw_counter_grid(frame, counts, last_saved_label):
    # Draw dark semi-transparent sidebar for counts
    overlay = frame.copy()
    h, w = frame.shape[:2]
    panel_w = 175
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(frame, "SAMPLES (A-Z)", (w - panel_w + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    y_start = 42
    line_h = 16

    # 2 columns inside sidebar
    col1 = letters[:13]
    col2 = letters[13:]

    for idx, (l1, l2) in enumerate(zip(col1, col2)):
        y = y_start + (idx * line_h)

        c1 = (0, 255, 0) if counts[l1] >= 80 else (0, 165, 255)
        if l1 == last_saved_label:
            c1 = (255, 255, 0)
        cv2.putText(frame, f"{l1}:{counts[l1]:3d}", (w - panel_w + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c1, 1)

        c2 = (0, 255, 0) if counts[l2] >= 80 else (0, 165, 255)
        if l2 == last_saved_label:
            c2 = (255, 255, 0)
        cv2.putText(frame, f"{l2}:{counts[l2]:3d}", (w - panel_w + 90, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c2, 1)


def main():
    cap = cv2.VideoCapture(0)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    file_exists = os.path.isfile(CSV_PATH)

    counts = load_existing_counts(CSV_PATH)
    last_saved_label = None

    with open(CSV_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['label'] + [f'f{i}' for i in range(126)]  # 42 landmarks * 3
            writer.writerow(header)

        print("Vision2Braille Gesture Collector")
        print("  - Press ANY letter key A-Z to save a sample (both 'q' and 'w' now work!)")
        print("  - Press [ESC] to quit.")
        print(f"Current total samples: {sum(counts.values())}")

        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            hand_count = 0
            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Draw status on the left
            status_text = f"Hands: {hand_count}/2"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if hand_count > 0 else (0, 0, 255), 2)
            cv2.putText(frame, "Press A-Z to capture | ESC to quit", (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

            if last_saved_label:
                cv2.putText(frame, f"Saved: {last_saved_label} (#{counts[last_saved_label]})",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Draw 2-column sidebar counters
            draw_counter_grid(frame, counts, last_saved_label)

            cv2.imshow("Data Collection - Vision2Braille", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key to quit safely without colliding with 'q'
                break
            elif key in KEY_TO_LABEL:
                if results.multi_hand_landmarks:
                    label = KEY_TO_LABEL[key]
                    features = extract_features(results.multi_hand_landmarks, results.multi_handedness)
                    writer.writerow([label] + features)
                    f.flush()
                    counts[label] += 1
                    last_saved_label = label
                    print(f"Saved sample for '{label}' (total: {counts[label]})")
                else:
                    print("No hand detected — cannot capture sample.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
