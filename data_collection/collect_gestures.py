import argparse
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
                if row:
                    lbl = row[0]
                    counts[lbl] = counts.get(lbl, 0) + 1
    return counts


def draw_counter_grid(frame, counts, last_saved_label, custom_target=None):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    panel_w = 175
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    if custom_target:
        cv2.putText(frame, "TARGET SIGN", (w - panel_w + 10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cnt = counts.get(custom_target, 0)
        color = (0, 255, 0) if cnt >= 80 else (0, 165, 255)
        cv2.putText(frame, f"{custom_target}: {cnt}", (w - panel_w + 10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "SPACE/ENTER to cap", (w - panel_w + 10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        return

    cv2.putText(frame, "SAMPLES (A-Z)", (w - panel_w + 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    letters = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    y_start = 42
    line_h = 16

    col1 = letters[:13]
    col2 = letters[13:]

    for idx, (l1, l2) in enumerate(zip(col1, col2)):
        y = y_start + (idx * line_h)

        c1 = (0, 255, 0) if counts.get(l1, 0) >= 80 else (0, 165, 255)
        if l1 == last_saved_label:
            c1 = (255, 255, 0)
        cv2.putText(frame, f"{l1}:{counts.get(l1, 0):3d}", (w - panel_w + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c1, 1)

        c2 = (0, 255, 0) if counts.get(l2, 0) >= 80 else (0, 165, 255)
        if l2 == last_saved_label:
            c2 = (255, 255, 0)
        cv2.putText(frame, f"{l2}:{counts.get(l2, 0):3d}", (w - panel_w + 90, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c2, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Vision2Braille Gesture Collector")
    parser.add_argument("--label", help="Target specific letter or custom word sign (e.g. --label HELLO or --label D)")
    return parser.parse_args()


def main():
    args = parse_args()
    custom_target = args.label.strip().upper() if args.label else None

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
        if custom_target:
            print(f"  - Dedicated capture mode for sign: '{custom_target}'")
            print("  - Press [SPACE] or [ENTER] to save a sample.")
        else:
            print("  - Standard alphabet mode: press any key A-Z to save a sample.")
            print("  - (To record a full word sign, run: python collect_gestures.py --label WORD)")
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

            status_text = f"Hands: {hand_count}/2"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if hand_count > 0 else (0, 0, 255), 2)

            bottom_prompt = f"Sign '{custom_target}' & press SPACE/ENTER | ESC to quit" if custom_target else "Press A-Z to capture | ESC to quit"
            cv2.putText(frame, bottom_prompt, (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

            if last_saved_label:
                cv2.putText(frame, f"Saved: {last_saved_label} (#{counts[last_saved_label]})",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            draw_counter_grid(frame, counts, last_saved_label, custom_target)

            cv2.imshow("Data Collection - Vision2Braille", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC to quit
                break

            target_label = None
            if custom_target:
                if key in (32, 10, 13):  # SPACE or ENTER
                    target_label = custom_target
            elif key in KEY_TO_LABEL:
                target_label = KEY_TO_LABEL[key]

            if target_label:
                if results.multi_hand_landmarks:
                    features = extract_features(results.multi_hand_landmarks, results.multi_handedness)
                    writer.writerow([target_label] + features)
                    f.flush()
                    counts[target_label] = counts.get(target_label, 0) + 1
                    last_saved_label = target_label
                    print(f"Saved sample for '{target_label}' (total: {counts[target_label]})")
                else:
                    print("No hand detected — position your hands before pressing capture.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
