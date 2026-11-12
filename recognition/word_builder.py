import argparse
import os
import pickle
import sys
import time
from collections import Counter, deque

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))
from braille_serial import BrailleSerialClient, DEFAULT_BAUD
from features import BRAILLE_PATTERNS, draw_braille_cell, extract_features, predict_gesture

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

HOLD_DURATION = 1.5   # seconds each letter's pattern is "held"
PAUSE_DURATION = 0.3  # seconds between letters
AUTO_ADD_HOLD_SECS = 1.2  # hold time to auto-commit in hands-free mode


class BrailleSimulator:
    """Non-blocking frame-by-frame simulator for single-cell Braille output."""

    def __init__(self, hold_s: float = HOLD_DURATION, pause_s: float = PAUSE_DURATION):
        self.hold_s = hold_s
        self.pause_s = pause_s
        self.queue = []
        self.current_idx = 0
        self.state = "IDLE"  # IDLE, HOLD, PAUSE
        self.state_start = 0.0
        self.active_letter = None
        self.active_dots = []

    @property
    def is_active(self) -> bool:
        return self.state != "IDLE"

    def start(self, text: str):
        cleaned = [c for c in text.upper() if c in BRAILLE_PATTERNS]
        if not cleaned:
            return
        self.queue = cleaned
        self.current_idx = 0
        self._enter_hold(0)
        print(f"\n[SIMULATOR] Starting rendering for '{text}'")

    def _enter_hold(self, idx: int):
        self.current_idx = idx
        self.active_letter = self.queue[idx]
        self.active_dots = BRAILLE_PATTERNS.get(self.active_letter, [])
        self.state = "HOLD"
        self.state_start = time.monotonic()
        print(f"  [SIM] [{self.active_letter}] Raising dots {self.active_dots} (hold {self.hold_s:.1f}s)")

    def update(self):
        if self.state == "IDLE":
            return
        now = time.monotonic()
        elapsed = now - self.state_start

        if self.state == "HOLD" and elapsed >= self.hold_s:
            print(f"  [SIM] [{self.active_letter}] Releasing pins (pause {self.pause_s:.1f}s)")
            self.state = "PAUSE"
            self.active_dots = []
            self.state_start = now
        elif self.state == "PAUSE" and elapsed >= self.pause_s:
            next_idx = self.current_idx + 1
            if next_idx < len(self.queue):
                self._enter_hold(next_idx)
            else:
                print("[SIMULATOR] Completed rendering word.\n")
                self.state = "IDLE"
                self.active_letter = None
                self.active_dots = []


def parse_args():
    parser = argparse.ArgumentParser(description="Vision2Braille Word Builder")
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate (default: 115200).")
    parser.add_argument("--simulate", action="store_true", help="Force console simulation without serial hardware.")
    parser.add_argument("--auto-add", action="store_true", help="Enable hands-free auto-add mode on startup.")
    return parser.parse_args()


def main():
    args = parse_args()
    serial_client = BrailleSerialClient(baud=args.baud)
    simulator = BrailleSimulator()

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
    candidate_conf = 0.0
    word_buffer = []

    auto_add_enabled = args.auto_add
    candidate_start_time = None
    last_auto_added = None

    print("\nVision2Braille Word Builder Ready")
    print("Controls:")
    print("  [SPACE]     = confirm candidate letter into word")
    print("  [ENTER]     = add space between words")
    print("  [BACKSPACE] = remove last character")
    print("  [A]         = toggle hands-free auto-add mode")
    print("  [S]         = send word to ESP32 Braille cell (or simulator)")
    print("  [T]         = test dots 1..6 sequentially (ESP32 only)")
    print("  [X]         = stop / release solenoids immediately")
    print("  [C]         = clear current word")
    print("  [ESC / Q]   = quit\n")

    try:
        while True:
            # Poll hardware or update simulation state
            for msg in serial_client.poll():
                print(f"[ESP32] {msg}")
            simulator.update()

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
                current_pred, candidate_conf, _ = predict_gesture(clf, features)

            recent_preds.append(current_pred)

            if len(recent_preds) == STABILITY_WINDOW:
                most_common, count = Counter(recent_preds).most_common(1)[0]
                if most_common is not None and count >= STABILITY_WINDOW * 0.8:
                    if candidate_letter != most_common:
                        candidate_letter = most_common
                        candidate_start_time = time.monotonic()
                        last_auto_added = None
                else:
                    candidate_letter = None
                    candidate_start_time = None
                    last_auto_added = None

            # Hands-free auto-commit logic
            auto_progress = 0.0
            if auto_add_enabled and candidate_letter and candidate_start_time:
                held_time = time.monotonic() - candidate_start_time
                auto_progress = min(1.0, held_time / AUTO_ADD_HOLD_SECS)
                if held_time >= AUTO_ADD_HOLD_SECS and last_auto_added != candidate_letter:
                    word_buffer.append(candidate_letter)
                    last_auto_added = candidate_letter
                    candidate_start_time = time.monotonic()  # reset timer
                    print(f"[AUTO-ADD] Added '{candidate_letter}' -> word: {''.join(word_buffer)}")

            # --- Visual Overlays ---
            word_display = "".join(word_buffer) if word_buffer else "(empty)"
            hw_color = (0, 255, 0) if serial_client.connected else (0, 180, 255)

            # Header info
            cv2.putText(frame, f"Candidate: {candidate_letter or '...'} ({candidate_conf * 100:.0f}%)",
                        (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
            cv2.putText(frame, f"Word: {word_display}", (10, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 200, 0), 2)
            cv2.putText(frame, f"HW: {serial_client.status_text}", (10, 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, hw_color, 2)

            # Auto-add progress bar
            if auto_add_enabled:
                mode_str = f"AUTO-ADD [ON] {int(auto_progress * 100)}%"
                cv2.putText(frame, mode_str, (10, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (50, 220, 255), 1)
                bar_w = int(120 * auto_progress)
                cv2.rectangle(frame, (10, 132), (130, 138), (60, 60, 60), -1)
                cv2.rectangle(frame, (10, 132), (10 + bar_w, 138), (50, 220, 255), -1)

            # Draw visual Braille cell
            active_cell_letter = simulator.active_letter if simulator.is_active else candidate_letter
            cell_title = "SIM" if simulator.is_active else "DOTS"
            draw_braille_cell(frame, active_cell_letter, x=15, y=148, title=cell_title)

            # Bottom helper bar
            cv2.putText(frame, "SPACE=add  ENTER=space  BKSP=undo  A=auto  S=send  T=test  C=clear  Q=quit",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 210, 210), 1)

            cv2.imshow("Vision2Braille - Word Builder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == 32:  # SPACE -> add letter
                if candidate_letter:
                    word_buffer.append(candidate_letter)
                    print(f"Added '{candidate_letter}' -> word so far: {''.join(word_buffer)}")
            elif key in (10, 13):  # ENTER -> add space
                word_buffer.append(' ')
                print(f"Added space -> word so far: '{''.join(word_buffer)}'")
            elif key == 8:   # BACKSPACE
                if word_buffer:
                    removed = word_buffer.pop()
                    print(f"Removed '{removed}' -> word so far: {''.join(word_buffer)}")
            elif key == ord('a'):
                auto_add_enabled = not auto_add_enabled
                candidate_start_time = None
                print(f"Hands-free Auto-Add mode: {'ON' if auto_add_enabled else 'OFF'}")
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
                simulator.state = "IDLE"
            elif key == ord('s'):
                if word_buffer:
                    word = "".join(word_buffer)
                    if serial_client.connected:
                        if serial_client.send_word(word):
                            print(f"Sent word '{word}' to ESP32 Braille cell.")
                            word_buffer = []
                        else:
                            print("Serial send failed — running visual simulation.")
                            simulator.start(word)
                            word_buffer = []
                    else:
                        simulator.start(word)
                        word_buffer = []
                else:
                    print("Nothing to send — word buffer is empty.")
    finally:
        serial_client.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
