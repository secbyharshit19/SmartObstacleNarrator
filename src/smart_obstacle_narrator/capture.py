"""Latest-frame live capture; video files use sequential capture in the application."""
import threading
import time

class LatestCapture:
    def __init__(self, source, width: int, height: int):
        import cv2
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open camera {source}. Close other camera apps and check permissions.')
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.packet = None
        self.error = ''
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name='capture', daemon=True)
        self.thread.start()

    def _run(self):
        sequence = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    self.error = 'Camera disconnected or frame capture failed'
                    return
                sequence += 1
                with self.lock:
                    self.packet = (sequence, time.monotonic(), frame)
        finally:
            self.cap.release()

    def read(self, after: int):
        with self.lock:
            if self.packet is None or self.packet[0] <= after:
                return None
            return self.packet

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=2)
