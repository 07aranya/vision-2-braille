# Vision2Braille

Real-time Indian Sign Language (ISL) fingerspelling recognition translated into sequential 6-dot Braille output for an ESP32-controlled push-solenoid Braille cell.

## Project Structure

```text
Vision2Braille/
├── data/
│   └── gestures.csv              # 2,376 labeled 126-D hand landmark samples (24 letters: A-Z except Q, W)
├── data_collection/
│   └── collect_gestures.py       # Webcam landmark capture & CSV dataset builder
├── model/
│   ├── train_classifier.py       # 5-NN classifier training & confusion matrix generation
│   ├── classifier.pkl            # Trained KNeighborsClassifier model
│   └── confusion_matrix.png      # Stratified 80/20 test confusion matrix
├── recognition/
│   ├── live_recognize.py         # Real-time single-letter gesture recognition
│   ├── word_builder.py           # Word composer with ESP32 serial & simulation output
│   └── braille_serial.py         # Non-blocking USB Serial client for the ESP32
├── firmware/
│   └── braille_cell.ino          # ESP32 firmware driving 6 solenoids via a ULN2803
├── dashboard/
│   ├── app.py                    # Web dashboard (placeholder)
│   └── templates/
└── debug_csv.py                  # Dataset label validation utility
```

## How It Works

1. **Hand Landmark Extraction:** MediaPipe Hands tracks up to 2 hands per frame and constructs a 126-element wrist-normalized feature vector (63 values for Left hand + 63 values for Right hand, zero-padded if a hand is absent).
2. **Classification:** A `KNeighborsClassifier(n_neighbors=5)` predicts the letter, stabilized across a 15-frame sliding window (`80%` consensus threshold).
3. **Word Building & Serial Dispatch:** Letters are confirmed into a word buffer in `recognition/word_builder.py` and sent over USB Serial (`115200` baud) to the ESP32 (`WORD:<TEXT>\n`). If no ESP32 is connected, it falls back to console simulation.
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
pip install opencv-python mediapipe scikit-learn pandas numpy matplotlib pyserial
```

### 2. Flash the ESP32 Firmware
Open `firmware/braille_cell.ino` in the Arduino IDE, select your ESP32 board and COM port, and upload.

### 3. Run the Word Builder
```powershell
python recognition/word_builder.py
```
Optional flags:
- `--port COM3` — specify the ESP32 serial port explicitly (auto-detected if omitted)
- `--baud 115200` — serial baud rate (default: `115200`)
- `--simulate` — force console simulation without serial hardware

**Word Builder Controls:**
- `SPACE` — confirm candidate letter into word
- `BACKSPACE` — remove last letter
- `S` — send word to ESP32 Braille cell (or simulate if disconnected)
- `T` — pulse solenoid dots `1`–`6` sequentially to test hardware wiring
- `X` — immediately stop and release all solenoids
- `C` — clear word buffer
- `Q` — quit

### 4. Collect More Data & Retrain (Optional)
```powershell
python data_collection/collect_gestures.py
python model/train_classifier.py
```
*(Note: In `collect_gestures.py`, `q` is bound to quit, and `Q` and `W` are currently not present in `data/gestures.csv`.)*
