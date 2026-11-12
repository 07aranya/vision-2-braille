/*
 * Vision2Braille - ESP32 Single-Cell Braille Actuator Firmware
 * ------------------------------------------------------------
 * Drives 6 push-type solenoids via a ULN2803 Darlington transistor array.
 * ULN2803 inputs are active-HIGH:
 *   GPIO HIGH -> ULN2803 output sinks current -> solenoid raises dot
 *   GPIO LOW  -> ULN2803 output open          -> solenoid releases dot
 *
 * Standard Braille Cell Layout:
 *   [Dot 1]  [Dot 4]
 *   [Dot 2]  [Dot 5]
 *   [Dot 3]  [Dot 6]
 *
 * Default ESP32 GPIO -> ULN2803 Input Mapping (safe output-capable pins):
 *   Dot 1 -> GPIO 13 -> ULN2803 IN1 (OUT1 -> Solenoid 1)
 *   Dot 2 -> GPIO 14 -> ULN2803 IN2 (OUT2 -> Solenoid 2)
 *   Dot 3 -> GPIO 27 -> ULN2803 IN3 (OUT3 -> Solenoid 3)
 *   Dot 4 -> GPIO 26 -> ULN2803 IN4 (OUT4 -> Solenoid 4)
 *   Dot 5 -> GPIO 25 -> ULN2803 IN5 (OUT5 -> Solenoid 5)
 *   Dot 6 -> GPIO 33 -> ULN2803 IN6 (OUT6 -> Solenoid 6)
 *   Note: Connect ULN2803 COM (pin 10) to solenoid +V supply for flyback protection,
 *         and share GND between ESP32, ULN2803 (pin 9), and solenoid power supply.
 *
 * Serial Protocol (115200 baud, newline-terminated):
 *   WORD:<TEXT>  - Render letters A-Z (and space) sequentially on the Braille cell
 *   <TEXT>       - Shortcut: any plain text line renders as a word (handy in Serial Monitor)
 *   STOP         - Immediately release all solenoids and cancel the current word
 *   TEST         - Sequentially pulse dots 1..6 to verify wiring
 *   PING         - Replies with PONG
 */

#include <Arduino.h>

// -------------------- Configuration --------------------
constexpr uint32_t SERIAL_BAUD = 115200;

// GPIO pins for Braille dots 1..6 (indices 0..5)
constexpr uint8_t DOT_PINS[6] = {13, 14, 27, 26, 25, 33};

// Timing in milliseconds (matches word_builder.py defaults)
constexpr uint32_t HOLD_DURATION_MS  = 1500;  // Time dots stay raised per letter
constexpr uint32_t PAUSE_DURATION_MS = 300;   // Cooldown / gap with all dots lowered
constexpr uint32_t TEST_STEP_MS      = 350;   // Duration per dot during TEST command

constexpr size_t MAX_WORD_LEN = 64;
constexpr size_t CMD_BUF_LEN  = 96;

// Bitmask for A-Z: bit 0 = Dot 1 ... bit 5 = Dot 6
constexpr uint8_t BRAILLE_ALPHABET[26] = {
    0x01, // A: {1}
    0x03, // B: {1,2}
    0x09, // C: {1,4}
    0x19, // D: {1,4,5}
    0x11, // E: {1,5}
    0x0B, // F: {1,2,4}
    0x1B, // G: {1,2,4,5}
    0x13, // H: {1,2,5}
    0x0A, // I: {2,4}
    0x1A, // J: {2,4,5}
    0x05, // K: {1,3}
    0x07, // L: {1,2,3}
    0x0D, // M: {1,3,4}
    0x1D, // N: {1,3,4,5}
    0x15, // O: {1,3,5}
    0x0F, // P: {1,2,3,4}
    0x1F, // Q: {1,2,3,4,5}
    0x17, // R: {1,2,3,5}
    0x0E, // S: {2,3,4}
    0x1E, // T: {2,3,4,5}
    0x25, // U: {1,3,6}
    0x27, // V: {1,2,3,6}
    0x3A, // W: {2,4,5,6}
    0x2D, // X: {1,3,4,6}
    0x3D, // Y: {1,3,4,5,6}
    0x35  // Z: {1,3,5,6}
};

// -------------------- State Machine --------------------
enum class RenderState {
    IDLE,
    WORD_HOLD,
    WORD_PAUSE,
    TEST_STEP
};

RenderState state = RenderState::IDLE;
char wordQueue[MAX_WORD_LEN + 1] = {0};
size_t wordLen = 0;
size_t wordIdx = 0;
uint8_t testDotIdx = 0;
uint32_t stateStartMs = 0;

char cmdBuf[CMD_BUF_LEN];
size_t cmdLen = 0;

// -------------------- Hardware Helpers --------------------
void releaseAllDots() {
    for (uint8_t i = 0; i < 6; ++i) {
        digitalWrite(DOT_PINS[i], LOW);
    }
}

void setDotMask(uint8_t mask) {
    for (uint8_t i = 0; i < 6; ++i) {
        digitalWrite(DOT_PINS[i], (mask & (1U << i)) ? HIGH : LOW);
    }
}

uint8_t getCharMask(char ch) {
    if (ch >= 'A' && ch <= 'Z') {
        return BRAILLE_ALPHABET[ch - 'A'];
    }
    return 0x00;  // Space or unsupported char -> all dots lowered
}

void formatMaskDots(uint8_t mask, char* out, size_t outSize) {
    if (outSize < 2) return;
    size_t pos = 0;
    for (uint8_t i = 0; i < 6; ++i) {
        if (mask & (1U << i)) {
            if (pos > 0 && pos < outSize - 1) {
                out[pos++] = ',';
            }
            if (pos < outSize - 1) {
                out[pos++] = static_cast<char>('1' + i);
            }
        }
    }
    if (pos == 0 && outSize >= 5) {
        strcpy(out, "none");
        return;
    }
    out[pos] = '\0';
}

void stopRendering(bool emitAck) {
    releaseAllDots();
    state = RenderState::IDLE;
    wordLen = 0;
    wordIdx = 0;
    wordQueue[0] = '\0';
    if (emitAck) {
        Serial.println("ACK:STOP");
    }
}

void startLetterAt(size_t idx) {
    char ch = wordQueue[idx];
    uint8_t mask = getCharMask(ch);
    setDotMask(mask);

    char dotsStr[16];
    formatMaskDots(mask, dotsStr, sizeof(dotsStr));
    Serial.print("DOT:");
    Serial.print(ch);
    Serial.print(":");
    Serial.println(dotsStr);

    state = RenderState::WORD_HOLD;
    stateStartMs = millis();
}

void startWord(const char* rawText) {
    stopRendering(false);

    size_t len = 0;
    for (size_t i = 0; rawText[i] != '\0' && len < MAX_WORD_LEN; ++i) {
        char c = rawText[i];
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 'a' + 'A');
        }
        if ((c >= 'A' && c <= 'Z') || c == ' ') {
            wordQueue[len++] = c;
        }
    }
    wordQueue[len] = '\0';
    wordLen = len;
    wordIdx = 0;

    if (wordLen == 0) {
        Serial.println("ERR:EMPTY_WORD");
        return;
    }

    Serial.print("ACK:WORD:");
    Serial.println(wordQueue);
    startLetterAt(0);
}

void startDotTest() {
    stopRendering(false);
    testDotIdx = 0;
    setDotMask(1U << testDotIdx);
    Serial.println("ACK:TEST");
    Serial.println("TEST_DOT:1");
    state = RenderState::TEST_STEP;
    stateStartMs = millis();
}

void handleCommand(char* line) {
    // Trim leading/trailing whitespace
    while (*line == ' ' || *line == '\t' || *line == '\r') {
        ++line;
    }
    size_t len = strlen(line);
    while (len > 0 && (line[len - 1] == ' ' || line[len - 1] == '\t' || line[len - 1] == '\r')) {
        line[--len] = '\0';
    }
    if (len == 0) {
        return;
    }

    if (strcasecmp(line, "PING") == 0) {
        Serial.println("PONG");
    } else if (strcasecmp(line, "STOP") == 0) {
        stopRendering(true);
    } else if (strcasecmp(line, "TEST") == 0) {
        startDotTest();
    } else if (strncasecmp(line, "WORD:", 5) == 0) {
        startWord(line + 5);
    } else {
        // Allow typing a plain word directly from Serial Monitor
        startWord(line);
    }
}

void pollSerial() {
    while (Serial.available() > 0) {
        char c = static_cast<char>(Serial.read());
        if (c == '\n') {
            cmdBuf[cmdLen] = '\0';
            handleCommand(cmdBuf);
            cmdLen = 0;
        } else if (c != '\r') {
            if (cmdLen < CMD_BUF_LEN - 1) {
                cmdBuf[cmdLen++] = c;
            }
        }
    }
}

void updateStateMachine() {
    uint32_t now = millis();

    switch (state) {
        case RenderState::IDLE:
            break;

        case RenderState::WORD_HOLD:
            if (now - stateStartMs >= HOLD_DURATION_MS) {
                releaseAllDots();
                Serial.print("RELEASE:");
                Serial.println(wordQueue[wordIdx]);
                state = RenderState::WORD_PAUSE;
                stateStartMs = now;
            }
            break;

        case RenderState::WORD_PAUSE:
            if (now - stateStartMs >= PAUSE_DURATION_MS) {
                ++wordIdx;
                if (wordIdx < wordLen) {
                    startLetterAt(wordIdx);
                } else {
                    Serial.print("DONE:");
                    Serial.println(wordQueue);
                    stopRendering(false);
                }
            }
            break;

        case RenderState::TEST_STEP:
            if (now - stateStartMs >= TEST_STEP_MS) {
                ++testDotIdx;
                if (testDotIdx < 6) {
                    setDotMask(1U << testDotIdx);
                    Serial.print("TEST_DOT:");
                    Serial.println(testDotIdx + 1);
                    stateStartMs = now;
                } else {
                    releaseAllDots();
                    Serial.println("DONE:TEST");
                    state = RenderState::IDLE;
                }
            }
            break;
    }
}

// -------------------- Arduino Entry Points --------------------
void setup() {
    for (uint8_t i = 0; i < 6; ++i) {
        pinMode(DOT_PINS[i], OUTPUT);
        digitalWrite(DOT_PINS[i], LOW);  // Ensure all solenoids start off
    }

    Serial.begin(SERIAL_BAUD);
    delay(200);
    Serial.println("READY:Vision2Braille");
}

void loop() {
    pollSerial();
    updateStateMachine();
}
