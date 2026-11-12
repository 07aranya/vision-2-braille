import os
import pickle
import sys
from collections import Counter, deque

import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(__file__))
from features import draw_braille_cell, extract_features, predict_gesture
from tts import TTSPlayer

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


def main():
    cap = cv2.VideoCapture(0)
    tts = TTSPlayer(enabled=True)
    STABILITY_WINDOW = 15   # require consistent prediction across this many frames
    recent_preds = deque(maxlen=STABILITY_WINDOW)
    confirmed_letter = None
    last_confidence = 0.0

    print("Starting live recognition. Controls: [V] = toggle voice, [Q / ESC] = quit.")

    try:
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
                current_pred, last_confidence, _ = predict_gesture(clf, features)
            else:
                last_confidence = 0.0

            recent_preds.append(current_pred)

            if len(recent_preds) == STABILITY_WINDOW:
                most_common, count = Counter(recent_preds).most_common(1)[0]
                if most_common is not None and count >= STABILITY_WINDOW * 0.8:
                    if most_common != confirmed_letter:
                        confirmed_letter = most_common
                        tts.speak_letter(confirmed_letter)
                        print(f"Recognized: {confirmed_letter} ({last_confidence * 100:.0f}%)")
                elif most_common is None and count >= STABILITY_WINDOW * 0.8:
                    confirmed_letter = None

            display_text = confirmed_letter if confirmed_letter else "..."
            tts_status = "ON" if tts.enabled else "OFF"
            cv2.putText(frame, f"Letter: {display_text} ({last_confidence * 100:.0f}%)  [Voice (V): {tts_status}]",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)
            draw_braille_cell(frame, confirmed_letter, x=15, y=60, title="DOTS")

            cv2.imshow("Vision2Braille - Live Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('v'):
                is_on = tts.toggle()
                print(f"Voice TTS: {'ON' if is_on else 'OFF'}")
                if is_on:
                    tts.say("voice on")
    finally:
        tts.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
