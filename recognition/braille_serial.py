import time
from typing import Callable, List, Optional

import serial
from serial.tools import list_ports


DEFAULT_BAUD = 115200
USB_UART_KEYWORDS = (
    "cp210",
    "ch340",
    "ch341",
    "ch910",
    "ftdi",
    "silicon labs",
    "espressif",
    "usb serial",
    "uart",
    "jtag",
)


def _is_candidate_usb_port(port_info) -> bool:
    """Return True if a serial port looks like a USB-UART or ESP32 bridge."""
    desc = (port_info.description or "").lower()
    hwid = (port_info.hwid or "").lower()
    if "bluetooth" in desc or "bthenum" in hwid:
        return False
    return any(kw in desc or kw in hwid for kw in USB_UART_KEYWORDS) or "usb" in hwid


class BrailleSerialClient:
    """Non-blocking USB Serial client for the ESP32 Braille cell firmware."""

    def __init__(self, baud: int = DEFAULT_BAUD, serial_factory: Callable[..., serial.Serial] = serial.Serial):
        self.baud = baud
        self._serial_factory = serial_factory
        self._ser: Optional[serial.Serial] = None
        self.port: Optional[str] = None
        self.is_rendering: bool = False
        self.status_text: str = "Not connected"
        self._rx_buffer: str = ""

    @property
    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def connect(self, port: Optional[str] = None, handshake_timeout: float = 2.5) -> bool:
        """
        Connect to the ESP32 Braille controller and verify with PING/PONG.
        If port is None, probes detected USB-UART serial ports automatically.
        """
        candidates = [port] if port else [p.device for p in list_ports.comports() if _is_candidate_usb_port(p)]
        if not candidates:
            self.status_text = "No USB serial port found (simulation mode)"
            return False

        for candidate in candidates:
            if self._try_connect_port(candidate, handshake_timeout):
                return True

        self.status_text = "ESP32 not responding (simulation mode)"
        return False

    def _try_connect_port(self, port: str, handshake_timeout: float) -> bool:
        self._rx_buffer = ""
        ser = None
        try:
            ser = self._serial_factory(
                port=port,
                baudrate=self.baud,
                timeout=0.05,
                write_timeout=0.5,
            )
            # Clear stale startup bytes and allow brief settle if DTR reset occurred
            time.sleep(0.2)
            ser.reset_input_buffer()

            deadline = time.monotonic() + handshake_timeout
            while time.monotonic() < deadline:
                ser.write(b"PING\n")
                ser.flush()
                time.sleep(0.15)
                lines = self._read_available_lines(ser)
                if any(line == "PONG" or line.startswith("READY:") for line in lines):
                    self._ser = ser
                    self.port = port
                    self.is_rendering = False
                    self.status_text = f"Connected ({port})"
                    return True

            ser.close()
            return False
        except (serial.SerialException, OSError):
            if ser is not None and ser.is_open:
                ser.close()
            return False

    def _read_available_lines(self, ser: serial.Serial) -> List[str]:
        waiting = getattr(ser, "in_waiting", 0)
        if not waiting:
            return []
        chunk = ser.read(waiting).decode("ascii", errors="ignore")
        self._rx_buffer += chunk
        if "\n" not in self._rx_buffer:
            return []
        parts = self._rx_buffer.split("\n")
        self._rx_buffer = parts.pop()
        return [line.strip() for line in parts if line.strip()]

    def send_word(self, word: str) -> bool:
        """Send a word to the ESP32 for sequential Braille rendering."""
        sanitized = "".join(c for c in word.upper() if ("A" <= c <= "Z") or c == " ")
        if not sanitized or not self.connected:
            return False
        return self._write_line(f"WORD:{sanitized}")

    def send_stop(self) -> bool:
        """Immediately release all solenoids and cancel rendering."""
        if not self.connected:
            return False
        ok = self._write_line("STOP")
        if ok:
            self.is_rendering = False
            self.status_text = f"Stopped ({self.port})"
        return ok

    def send_test(self) -> bool:
        """Trigger sequential 1..6 solenoid test on the ESP32."""
        if not self.connected:
            return False
        return self._write_line("TEST")

    def _write_line(self, payload: str) -> bool:
        if not self.connected:
            return False
        try:
            self._ser.write(f"{payload}\n".encode("ascii"))
            self._ser.flush()
            return True
        except (serial.SerialException, OSError) as exc:
            self.status_text = f"Serial error: {exc}"
            self.close()
            return False

    def poll(self) -> List[str]:
        """Read and process any pending status lines from the ESP32."""
        if not self.connected:
            return []
        try:
            lines = self._read_available_lines(self._ser)
        except (serial.SerialException, OSError) as exc:
            self.status_text = f"Serial disconnected: {exc}"
            self.close()
            return []

        for line in lines:
            self._update_status_from_line(line)
        return lines

    def _update_status_from_line(self, line: str) -> None:
        if line.startswith("ACK:WORD:"):
            self.is_rendering = True
            self.status_text = f"Sending: {line[9:]}"
        elif line.startswith("DOT:"):
            parts = line.split(":", 2)
            if len(parts) == 3:
                self.is_rendering = True
                self.status_text = f"Braille [{parts[1]}] dots {parts[2]}"
        elif line.startswith("RELEASE:"):
            self.status_text = f"Released [{line[8:]}]"
        elif line.startswith("DONE:"):
            self.is_rendering = False
            self.status_text = f"Done ({self.port})"
        elif line == "ACK:STOP":
            self.is_rendering = False
            self.status_text = f"Stopped ({self.port})"
        elif line.startswith("TEST_DOT:"):
            self.is_rendering = True
            self.status_text = f"Testing dot {line[9:]}"

    def close(self) -> None:
        """Release solenoids if active and close the serial port."""
        if self._ser is not None:
            try:
                if self._ser.is_open:
                    if self.is_rendering:
                        self._ser.write(b"STOP\n")
                        self._ser.flush()
                    self._ser.close()
            except (serial.SerialException, OSError):
                pass
            finally:
                self._ser = None
                self.is_rendering = False
                self._rx_buffer = ""
