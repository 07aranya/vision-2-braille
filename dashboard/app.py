import argparse
import os
import pickle
import sys
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, render_template, request

# Ensure Vision2Braille/recognition is importable regardless of working directory
RECOGNITION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recognition'))
sys.path.insert(0, RECOGNITION_DIR)

from braille_serial import BrailleSerialClient, DEFAULT_BAUD
from features import BRAILLE_PATTERNS, extract_features, predict_gesture
from word_builder import AUTO_ADD_HOLD_SECS, BrailleSimulator

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'classifier.pkl'))

app = Flask(__name__)


class DashboardEngine:
    """Thread-safe webcam recognition, word builder, and Braille cell controller."""

    def __init__(self, port=None, baud=DEFAULT_BAUD, simulate=False):
        self.lock = threading.Lock()
        self.serial_client = BrailleSerialClient(baud=baud)
        self.simulator = BrailleSimulator()
        self.logs = deque(maxlen=60)

        if simulate:
            self.serial_client.status_text = "Simulation mode (--simulate)"
            self._log("SYS", "Started in simulation mode (--simulate).")
        else:
            if self.serial_client.connect(port=port):
                self._log("SYS", f"Connected to ESP32 on {self.serial_client.port}.")
            else:
                self._log("SYS", f"ESP32 not found ({self.serial_client.status_text}). Using simulator.")

        with open(MODEL_PATH, 'rb') as f:
            self.clf = pickle.load(f)

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        self.stability_window = 15
        self.recent_preds = deque(maxlen=self.stability_window)
        self.candidate = None
        self.confidence = 0.0
        self.hand_count = 0
        self.fps = 0.0
        self.word_buffer = []

        self.auto_add = False
        self.candidate_since = None
        self.last_auto_added = None
        self.auto_progress = 0.0

        self.hw_active_letter = None
        self.hw_active_dots = []
        self.rendering_word = ""
        self.rendering_idx = -1

        self.latest_jpeg = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _log(self, tag: str, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append({"time": timestamp, "tag": tag, "msg": msg})

    def _handle_esp_line(self, line: str):
        self._log("RX", line)
        if line.startswith("ACK:WORD:"):
            self.rendering_word = line[9:]
            self.rendering_idx = -1
        elif line.startswith("DOT:"):
            parts = line.split(":", 2)
            if len(parts) == 3:
                self.hw_active_letter = parts[1]
                self.rendering_idx += 1
                dots_raw = parts[2]
                if dots_raw == "none":
                    self.hw_active_dots = []
                else:
                    self.hw_active_dots = [int(d) for d in dots_raw.split(",") if d.isdigit()]
        elif line.startswith("TEST_DOT:"):
            dot_str = line.split(":", 1)[1]
            if dot_str.isdigit():
                self.hw_active_letter = f"T{dot_str}"
                self.hw_active_dots = [int(dot_str)]
        elif line.startswith("RELEASE:"):
            self.hw_active_letter = None
            self.hw_active_dots = []
        elif line.startswith("DONE:") or line == "ACK:STOP":
            self.hw_active_letter = None
            self.hw_active_dots = []
            self.rendering_word = ""
            self.rendering_idx = -1

    def _loop(self):
        cap = cv2.VideoCapture(0)
        prev_time = time.monotonic()
        try:
            while self.running:
                with self.lock:
                    for msg in self.serial_client.poll():
                        self._handle_esp_line(msg)
                    prev_sim_state = self.simulator.state
                    prev_sim_idx = self.simulator.current_idx
                    self.simulator.update()
                    if self.simulator.is_active:
                        self.rendering_word = "".join(self.simulator.queue)
                        self.rendering_idx = self.simulator.current_idx
                        if prev_sim_state != self.simulator.state or prev_sim_idx != self.simulator.current_idx:
                            if self.simulator.state == "HOLD":
                                dots = ",".join(str(d) for d in self.simulator.active_dots) or "none"
                                self._log("SIM", f"DOT:{self.simulator.active_letter}:{dots}")
                            elif self.simulator.state == "PAUSE":
                                self._log("SIM", f"RELEASE:{self.simulator.active_letter}")
                    elif prev_sim_state != "IDLE":
                        self._log("SIM", f"DONE:{self.rendering_word}")
                        self.rendering_word = ""
                        self.rendering_idx = -1

                success, frame = cap.read()
                if not success:
                    time.sleep(0.05)
                    continue

                now = time.monotonic()
                dt = now - prev_time
                prev_time = now
                inst_fps = (1.0 / dt) if dt > 0 else 0.0

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb)

                pred = None
                conf = 0.0
                detected_hands = 0
                if results.multi_hand_landmarks:
                    detected_hands = len(results.multi_hand_landmarks)
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_draw.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_styles.get_default_hand_landmarks_style(),
                            self.mp_styles.get_default_hand_connections_style(),
                        )
                    features = extract_features(results.multi_hand_landmarks, results.multi_handedness)
                    pred, conf, _ = predict_gesture(self.clf, features)

                with self.lock:
                    self.fps = self.fps * 0.8 + inst_fps * 0.2
                    self.hand_count = detected_hands
                    self.recent_preds.append(pred)

                    if pred is not None:
                        self.confidence = conf
                    elif detected_hands == 0:
                        self.confidence = 0.0

                    if len(self.recent_preds) == self.stability_window:
                        most_common, count = Counter(self.recent_preds).most_common(1)[0]
                        if most_common is not None and count >= self.stability_window * 0.8:
                            if self.candidate != most_common:
                                self.candidate = most_common
                                self.candidate_since = time.monotonic()
                                self.last_auto_added = None
                        else:
                            self.candidate = None
                            self.candidate_since = None
                            self.last_auto_added = None

                    if self.auto_add and self.candidate and self.candidate_since:
                        held = time.monotonic() - self.candidate_since
                        self.auto_progress = min(1.0, held / AUTO_ADD_HOLD_SECS)
                        if held >= AUTO_ADD_HOLD_SECS and self.last_auto_added != self.candidate:
                            for c in self.candidate:
                                self.word_buffer.append(c)
                            self.last_auto_added = self.candidate
                            self.candidate_since = time.monotonic()
                            self._log("ADD", f"Auto-added '{self.candidate}'")
                    else:
                        self.auto_progress = 0.0

                ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    with self.lock:
                        self.latest_jpeg = encoded.tobytes()
        finally:
            cap.release()

    def get_state(self):
        with self.lock:
            if self.serial_client.connected and (self.hw_active_letter is not None or self.serial_client.is_rendering):
                active_letter = self.hw_active_letter
                active_dots = list(self.hw_active_dots)
                cell_mode = "HARDWARE"
            elif self.simulator.is_active:
                active_letter = self.simulator.active_letter
                active_dots = list(self.simulator.active_dots)
                cell_mode = "SIMULATOR"
            else:
                active_letter = self.candidate
                active_dots = BRAILLE_PATTERNS.get(self.candidate, []) if self.candidate else []
                cell_mode = "PREVIEW"

            return {
                "candidate": self.candidate,
                "confidence": round(self.confidence * 100),
                "hand_count": self.hand_count,
                "fps": round(self.fps, 1),
                "word": "".join(self.word_buffer),
                "rendering_word": self.rendering_word,
                "rendering_idx": self.rendering_idx,
                "hw_connected": self.serial_client.connected,
                "hw_port": self.serial_client.port,
                "hw_status": self.serial_client.status_text,
                "active_letter": active_letter,
                "active_dots": active_dots,
                "cell_mode": cell_mode,
                "auto_add": self.auto_add,
                "auto_progress": round(self.auto_progress * 100),
                "logs": list(self.logs),
                "classes": list(self.clf.classes_),
            }

    def add_letter(self, ch=None):
        with self.lock:
            letter = ch or self.candidate
            if letter:
                for c in letter:
                    self.word_buffer.append(c)
                self._log("ADD", f"Added '{letter}'")
                return True
            return False

    def backspace(self):
        with self.lock:
            if self.word_buffer:
                self.word_buffer.pop()

    def clear_word(self):
        with self.lock:
            self.word_buffer = []

    def toggle_auto_add(self):
        with self.lock:
            self.auto_add = not self.auto_add
            self.candidate_since = None
            self.auto_progress = 0.0
            self._log("SYS", f"Auto-Add {'enabled' if self.auto_add else 'disabled'}.")

    def send_text(self, custom_text=None):
        with self.lock:
            if custom_text is not None:
                raw = "".join(c for c in custom_text.upper() if c in BRAILLE_PATTERNS)
            else:
                raw = "".join(self.word_buffer)
                self.word_buffer = []

            if not raw:
                return False, "Text is empty."

            if self.serial_client.connected:
                if self.serial_client.send_word(raw):
                    self.rendering_word = raw
                    self.rendering_idx = -1
                    self._log("TX", f"WORD:{raw}")
                    return True, f"Sent '{raw}' to ESP32."

            self.simulator.start(raw)
            self.rendering_word = raw
            self.rendering_idx = 0
            self._log("TX", f"WORD:{raw} (Simulator)")
            return True, f"Simulating '{raw}'."

    def connect_hardware(self, port=None):
        with self.lock:
            self.serial_client.close()
            ok = self.serial_client.connect(port=port or None)
            self._log(
                "SYS",
                f"Connected to {self.serial_client.port}"
                if ok else f"Connect failed: {self.serial_client.status_text}"
            )
            return ok, self.serial_client.status_text

    def test_hardware(self):
        with self.lock:
            if self.serial_client.connected:
                self.serial_client.send_test()
                self._log("TX", "TEST")
                return True
            return False

    def ping_hardware(self):
        with self.lock:
            if self.serial_client.connected:
                self.serial_client._write_line("PING")
                self._log("TX", "PING")
                return True
            return False

    def stop_output(self):
        with self.lock:
            self.simulator.state = "IDLE"
            self.simulator.active_letter = None
            self.simulator.active_dots = []
            self.hw_active_letter = None
            self.hw_active_dots = []
            self.rendering_word = ""
            self.rendering_idx = -1
            if self.serial_client.connected:
                self.serial_client.send_stop()
                self._log("TX", "STOP")
            else:
                self._log("SIM", "STOP")

    def clear_logs(self):
        with self.lock:
            self.logs.clear()

    def shutdown(self):
        self.running = False
        self.serial_client.close()


engine = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with engine.lock:
                frame_bytes = engine.latest_jpeg
            if frame_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.03)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(engine.get_state())


@app.route("/api/word/add", methods=["POST"])
def api_word_add():
    ok = engine.add_letter()
    return jsonify({"ok": ok})


@app.route("/api/word/space", methods=["POST"])
def api_word_space():
    ok = engine.add_letter(" ")
    return jsonify({"ok": ok})


@app.route("/api/word/backspace", methods=["POST"])
def api_word_backspace():
    engine.backspace()
    return jsonify({"ok": True})


@app.route("/api/word/clear", methods=["POST"])
def api_word_clear():
    engine.clear_word()
    return jsonify({"ok": True})


@app.route("/api/word/send", methods=["POST"])
def api_word_send():
    payload = request.get_json(silent=True) or {}
    custom = payload.get("text")
    ok, msg = engine.send_text(custom_text=custom)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/auto_add", methods=["POST"])
def api_auto_add():
    engine.toggle_auto_add()
    return jsonify({"ok": True})


@app.route("/api/hardware/connect", methods=["POST"])
def api_hw_connect():
    payload = request.get_json(silent=True) or {}
    port = payload.get("port")
    ok, status = engine.connect_hardware(port=port)
    return jsonify({"ok": ok, "status": status})


@app.route("/api/hardware/test", methods=["POST"])
def api_hw_test():
    return jsonify({"ok": engine.test_hardware()})


@app.route("/api/hardware/ping", methods=["POST"])
def api_hw_ping():
    return jsonify({"ok": engine.ping_hardware()})


@app.route("/api/hardware/stop", methods=["POST"])
def api_hw_stop():
    engine.stop_output()
    return jsonify({"ok": True})


@app.route("/api/logs/clear", methods=["POST"])
def api_logs_clear():
    engine.clear_logs()
    return jsonify({"ok": True})


def parse_args():
    parser = argparse.ArgumentParser(description="Vision2Braille Local Web Dashboard")
    parser.add_argument("--port", help="ESP32 serial port (e.g. COM3)")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help="Serial baud rate (default: 115200)")
    parser.add_argument("--simulate", action="store_true", help="Force simulation mode")
    parser.add_argument("--host", default="127.0.0.1", help="Flask host (default: 127.0.0.1)")
    parser.add_argument("--web-port", type=int, default=5000, help="Web server port (default: 5000)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    engine = DashboardEngine(port=args.port, baud=args.baud, simulate=args.simulate)
    print(f"Dashboard running at http://{args.host}:{args.web_port}")
    try:
        app.run(host=args.host, port=args.web_port, debug=False, threaded=True)
    finally:
        engine.shutdown()
