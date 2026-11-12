# Vision2Braille

Real-time Indian Sign Language (ISL) fingerspelling recognition translated into sequential 6-dot Braille output for an ESP32-controlled push-solenoid Braille cell.

## Project Structure

```text
Vision2Braille/
├── data/
│   └── gestures.csv              # Labeled 126-D hand landmark samples
├── data_collection/
│   └── collect_gestures.py       # Webcam landmark capture (A-Z, ESC to quit)
├── model/
│   ├── train_classifier.py       # 5-NN classifier training & confusion matrix generation
│   ├── classifier.pkl            # Trained KNeighborsClassifier model
│   └── confusion_matrix.png      # Stratified 80/20 test confusion matrix
├── recognition/
│   ├── features.py               # Shared feature extraction, confidence gating & Braille drawing
│   ├── live_recognize.py         # Real-time single-letter gesture recognition
│   ├── word_builder.py           # OpenCV word composer (hands-free auto-add + ESP32/sim output)
│   └── braille_serial.py         # Non-blocking USB Serial client for the ESP32
├── dashboard/
│   ├── app.py                    # Local Flask web dashboard with live Braille visualizer
│   └── templates/index.html      # Dashboard UI & serial activity log
├── firmware/
│   └── braille_cell.ino          # ESP32 firmware driving 6 solenoids via a ULN2803
├── debug_csv.py                  # Dataset label validation utility
└── requirements.txt              # Python dependencies
```

## How It Works

1. **Hand Landmark Extraction:** MediaPipe Hands tracks up to 2 hands per frame and constructs a 126-element wrist-normalized feature vector (63 values for Left hand + 63 values for Right hand, zero-padded if a hand is absent).
2. **Classification & Filtering:** A `KNeighborsClassifier(n_neighbors=5)` predicts the letter, filtered by probability (`>= 60%`) and nearest-neighbor distance (`<= 2.2`) to reject resting/transition poses, then stabilized across a 15-frame sliding window (`80%` consensus threshold).
3. **Word Building & Serial Dispatch:** Letters are confirmed into a word buffer (manually or via hands-free Auto-Add hold) and sent over USB Serial (`115200` baud) to the ESP32 (`WORD:<TEXT>\n`). If no ESP32 is connected, it runs a non-blocking visual simulation.
4. **Braille Actuation:** `firmware/braille_cell.ino` maps each letter to its 6-dot Grade 1 Braille bitmask and actuates 6 push-type solenoids through a ULN2803 Darlington driver (`1.5 s` hold, `0.3 s` release pause between letters).

## Hardware Wiring (ESP32 + ULN2803 + 6 Push Solenoids)

ULN2803 inputs are active-HIGH (`GPIO HIGH` energizes/raises the solenoid; `GPIO LOW` releases it).

| Braille Dot | ESP32 GPIO | ULN2803 Input |
| :--- | :--- | :--- |
| Dot 1 (top-left) | `GPIO 13` | `IN1` |
| Dot 2 (mid-left) | `GPIO 14` | `IN2` |
| Dot 3 (bot-left) | `GPIO 27` | `IN3` |
| Dot 4 (top-right) | `GPIO 26` | `IN4` |
| Dot 5 (mid-right) | `GPIO 25` | `IN5` |
| Dot 6 (bot-right) | `GPIO 33` | `IN6` |

- Connect **ULN2803 pin 10 (`COM`)** to the solenoid `+V` supply for flyback diode protection.
- Tie **ESP32 `GND`**, **ULN2803 pin 9 (`GND`)**, and the **solenoid power supply `GND`** together.

## Usage

### 1. Install Python Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Collect Missing Letters (`Q`, `W`) & Retrain
```powershell
python data_collection/collect_gestures.py
python model/train_classifier.py
```
- Press any letter key `A`–`Z` (including `Q` and `W`) to capture a sample; press `ESC` to quit.

### 3. Flash the ESP32 Firmware
Open `firmware/braille_cell.ino` in the Arduino IDE, select your ESP32 board and COM port, and upload.

### 4. Run the Local Web Dashboard (Recommended for Demo)
```powershell
python dashboard/app.py
```
Open `http://127.0.0.1:5000` in your browser for the live webcam feed, 6-dot Braille visualizer, word builder, and ESP32 serial console.

### 5. Or Run the OpenCV Word Builder
```powershell
python recognition/word_builder.py
```
Optional flags:
- `--port COM3` — specify the ESP32 serial port explicitly (auto-detected if omitted)
- `--baud 115200` — serial baud rate (default: `115200`)
- `--simulate` — force non-blocking simulation without serial hardware
- `--auto-add` — enable hands-free hold-to-confirm mode on startup

**Controls:** `SPACE` add letter · `ENTER` add space · `BACKSPACE` undo · `A` toggle auto-add · `S` send word · `T` test dots 1..6 · `X` stop · `C` clear · `Q`/`ESC` quit
