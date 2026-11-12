"""Non-blocking Text-to-Speech (TTS) module for Vision2Braille.

Provides asynchronous spoken feedback for confirmed letters and completed words
using pyttsx3 without blocking real-time OpenCV video capture or MediaPipe loops.
"""

import os
import queue
import sys
import threading
import time
from typing import Optional


class TTSPlayer:
    """Thread-safe, non-blocking Text-to-Speech engine."""

    def __init__(self, enabled: bool = True, rate: int = 175, volume: float = 1.0):
        self.enabled = enabled
        self.rate = rate
        self.volume = volume

        self._queue = queue.Queue(maxsize=12)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._engine = None

        if self.enabled:
            self._start_worker()

    def _start_worker(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TTS-Worker")
        self._thread.start()

    def _worker_loop(self):
        # On Windows, pyttsx3 (SAPI5) requires COM initialization on background threads
        if sys.platform == "win32":
            try:
                import pythoncom
                pythoncom.CoInitialize()
            except ImportError:
                pass

        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self.rate)
            self._engine.setProperty('volume', self.volume)
        except Exception as e:
            print(f"[TTS] Warning: Failed to initialize pyttsx3 engine: {e}")
            self._engine = None
            return

        while self._running:
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if text is None:
                self._queue.task_done()
                break

            if self.enabled and self._engine:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception as e:
                    print(f"[TTS] Speech error on '{text}': {e}")

            self._queue.task_done()

    def say(self, text: str, clear_pending: bool = False):
        """Enqueue arbitrary text to be spoken."""
        if not self.enabled or not text:
            return

        if not self._running:
            self._start_worker()

        if clear_pending:
            # Drop older pending utterances so speech stays up-to-date
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except (queue.Empty, ValueError):
                    break

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass  # Avoid unbounded queue backlog

    def speak_letter(self, letter: str):
        """Speak a single confirmed letter clearly."""
        if not letter or not self.enabled:
            return
        letter = letter.strip().upper()
        if not letter:
            return
        # Clarify single vowels or letters if needed, otherwise speak character
        self.say(letter, clear_pending=True)

    def speak_word(self, word: str):
        """Speak a completed word."""
        if not word or not self.enabled:
            return
        cleaned = word.strip()
        if cleaned:
            self.say(cleaned, clear_pending=True)

    def toggle(self) -> bool:
        """Toggle TTS on/off and return new state."""
        self.enabled = not self.enabled
        if self.enabled and not self._running:
            self._start_worker()
        return self.enabled

    def stop(self):
        """Stop worker and release engine resources."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._engine = None
