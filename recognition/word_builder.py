import argparse
import cv2
import mediapipe as mp
import pickle
import os
import time
from collections import deque, Counter

from braille_serial import BrailleSerialClient, DEFAULT_BAUD

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

# --- Braille dot patterns (for demo/simulation display only) ---
BRAILLE_PATTERNS = {
    'A': [1], 'B': [1,2], 'C': [1,4], 'D': [1,4,5], 'E': [1,5],
    'F': [1,2,4], 'G': [1,2,4,5], 'H': [1,2,5], 'I': [2,4], 'J': [2,4,5],
    'K': [1,3], 'L': [1,2,3], 'M': [1,3,4], 'N': [1,3,4,5], 'O': [1,3,5],
    'P': [1,2,3,4], 'Q': [1,2,3,4,5], 'R': [1,2,3,5], 'S': [2,3,4], 'T': [2,3,4,5],
    'U': [1,3,6], 'V': [1,2,3,6], 'W': [2,4,5,6], 'X': [1,3,4,6], 'Y': [1,3,4,5,6], 'Z': [1,3,5,6],
}

HOLD_DURATION = 1.5   # seconds each letter's pattern is "held"
PAUSE_DURATION = 0.3  # seconds between letters


def extract_features(multi_hand_landmarks, multi_handedness):
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
        label = handedness.classification[0].label
        feats = normalize(hand_landmarks)
        if label == 'Left':
            left_feats = feats
        else:
            right_feats = feats

    return left_feats + right_feats


def simulate_braille_output(word):
    """
    Simulates the sequential single-cell Braille actuation.
    Prints the dot pattern for each letter with the actual
    hold/pause timing that runs on the ESP32 firmware.
    """
    print("\n" + "=" * 40)
    print(f"SENDING TO BRAILLE CELL (SIMULATION): '{word}'")
    print("=" * 40)
    for letter in word:
        pattern = BRAILLE_PATTERNS.get(letter, [])
        print(f"  [{letter}] Raising dots {pattern} ... (hold {HOLD_DURATION}s)")
        time.sleep(HOLD_DURATION)
        print(f"  [{letter}] Releasing all pins ... (pause {PAUSE_DURATION}s)")
        time.sleep(PAUSE_DURATION)
    print("=" * 40)
    print("Word fully rendered on Braille cell.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Vision2Braille Word Builder")
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate (default: 115200).")
    parser.add_argument("--simulate", action="store_true", help="Force console simulation without serial hardware.")
    return parser.parse_args()


def main():
    args = parse_args()
    serial_client = BrailleSerialClient(baud=args.baud)

    if args.simulate:
        serial_client.status_text = "Simulation mode (--simulate)"
        print("Running in simulation mode (--simulate).")
    else:
        if serial_client.connect(port=args.port):
            print(f"Connected to ESP32 Braille cell on {serial_client.port}.")
        else:
            print(f"ESP32 not detected ({serial_client.status_text}). Falling back to simulation.")

    cap = cv2.VideoCapture(0)
    STABILITY_WINDOW = 15
    recent_preds = deque(maxlen=STABILITY_WINDOW)
    candidate_letter = None
    word_buffer = []

    print("Controls:")
    print("  [SPACE] = confirm current letter into word")
    print("  [BACKSPACE] = remove last letter")
    print("  [S] = send word to Braille cell (or simulate if disconnected)")
    print("  [T] = test solenoid dots 1..6 (hardware only)")
    print("  [X] = stop/release solenoids immediately (hardware only)")
    print("  [C] = clear word")
    print("  [Q] = quit")

    try:
        while True:
            for msg in serial_client.poll():
                print(f"[ESP32] {msg}")

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

            if len(recent_preds) == STABILITY_WINDOW:
                most_common, count = Counter(recent_preds).most_common(1)[0]
                if most_common is not None and count >= STABILITY_WINDOW * 0.8:
                    candidate_letter = most_common
                else:
                    candidate_letter = None

            # --- On-screen overlay ---
            word_display = "".join(word_buffer) if word_buffer else "(empty)"
            hw_color = (0, 255, 0) if serial_client.connected else (0, 180, 255)
            cv2.putText(frame, f"Candidate: {candidate_letter or '...'}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Word: {word_display}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
            cv2.putText(frame, f"Hardware: {serial_client.status_text}", (10, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, hw_color, 2)
            cv2.putText(frame, "SPACE=add  BKSP=undo  S=send  T=test  X=stop  C=clear  Q=quit",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            cv2.imshow("Vision2Braille - Word Builder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 32:  # SPACE
                if candidate_letter:
                    word_buffer.append(candidate_letter)
                    print(f"Added '{candidate_letter}' -> word so far: {''.join(word_buffer)}")
            elif key == 8:   # BACKSPACE
                if word_buffer:
                    removed = word_buffer.pop()
                    print(f"Removed '{removed}' -> word so far: {''.join(word_buffer)}")
            elif key == ord('c'):
                word_buffer = []
                print("Word cleared.")
            elif key == ord('t'):
                if serial_client.connected:
                    serial_client.send_test()
                    print("Sent TEST command to ESP32.")
                else:
                    print("Hardware not connected — cannot run solenoid test.")
            elif key == ord('x'):
                if serial_client.connected:
                    serial_client.send_stop()
                    print("Sent STOP command to ESP32.")
            elif key == ord('s'):
                if word_buffer:
                    word = "".join(word_buffer)
                    if serial_client.connected:
                        if serial_client.send_word(word):
                            print(f"Sent word '{word}' to ESP32 Braille cell.")
                            word_buffer = []
                        else:
                            print("Failed to send over serial — falling back to simulation.")
                            simulate_braille_output(word)
                            word_buffer = []
                    else:
                        simulate_braille_output(word)
                        word_buffer = []
                else:
                    print("Nothing to save — word is empty.")
    finally:
        serial_client.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
