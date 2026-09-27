"""A single owning speech thread, bounded pending work, and start acknowledgments."""
import queue
import threading
import time
from typing import Callable
from .types import WarningEvent, SpeechFeedback

class PyttsxBackend:
    def __init__(self, rate: int = 175):
        import pyttsx3
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)

    def say(self, text: str, on_start: Callable[[], None]) -> None:
        errors = []
        token = self.engine.connect('started-utterance', lambda name: on_start())
        error_token = self.engine.connect('error', lambda name, exception: errors.append(str(exception)))
        try:
            self.engine.say(text)
            self.engine.runAndWait()
            if errors:
                raise RuntimeError('; '.join(errors))
        finally:
            self.engine.disconnect(token)
            self.engine.disconnect(error_token)

    def close(self) -> None:
        self.engine.stop()

class SilentBackend:
    """Simulation backend: emits timestamps but produces NO audible speech."""
    def say(self, text: str, on_start: Callable[[], None]) -> None:
        on_start()
    def close(self) -> None:
        pass

class SpeechWorker:
    def __init__(self, gap: float, freshness: float, backend_factory: Callable,
                 clock: Callable[[], float] = time.monotonic):
        self.gap, self.freshness, self.factory, self.clock = gap, freshness, backend_factory, clock
        self.feedback: queue.Queue[SpeechFeedback] = queue.Queue()
        self.condition = threading.Condition()
        self.pending: WarningEvent | None = None
        self.active: WarningEvent | None = None
        self.snapshot: dict[int, tuple[tuple[str,str],float]] = {}
        self.muted = False
        self.closed = False
        self.status = 'starting'
        self.error = ''
        self.last_finished = -float('inf')
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, name='speech', daemon=True)
        self.thread.start()

    def update(self, snapshot: dict, event: WarningEvent | None) -> None:
        with self.condition:
            self.snapshot = snapshot
            self.pending = (event if not self.muted and not self.closed and
                            not (self.active and event and event.track_id == self.active.track_id) else None)
            self.condition.notify_all()

    def set_muted(self, muted: bool) -> None:
        with self.condition:
            self.muted = muted
            self.pending = None
            self.condition.notify_all()

    def _valid(self, e: WarningEvent, now: float) -> bool:
        state = self.snapshot.get(e.track_id)
        return bool(state and state[0] == e.signature and now-state[1] <= self.freshness and now <= e.expires)

    def _run(self) -> None:
        backend = None
        try:
            backend = self.factory()
            self.status = 'ready'
            self.ready.set()
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.closed or (self.pending is not None and not self.muted))
                    if self.closed:
                        return
                    now = self.clock()
                    e = self.pending
                    if e is None:
                        continue
                    if not self._valid(e, now):
                        self.pending = None
                        self.feedback.put(SpeechFeedback(e, 'dropped', now, 'expired or no longer relevant'))
                        continue
                    wait = self.gap - (now-self.last_finished)
                    if wait > 0:
                        self.condition.wait(min(wait, 0.1))
                        continue
                    self.pending = None
                    self.active = e
                started = False
                def on_start():
                    nonlocal started
                    if not started:
                        started = True
                        self.feedback.put(SpeechFeedback(e, 'started', self.clock()))
                self.status = 'speaking'
                backend.say(e.text, on_start)
                if not started:
                    raise RuntimeError('Speech driver produced no start notification; check audio')
                self.last_finished = self.clock()
                self.feedback.put(SpeechFeedback(e, 'finished', self.last_finished))
                with self.condition:
                    self.active = None
                self.status = 'ready'
        except Exception as exc:
            self.error = str(exc)
            self.status = 'error'
            if self.active:
                self.feedback.put(SpeechFeedback(self.active, 'error', self.clock(), str(exc)))
        finally:
            self.ready.set()
            if backend:
                backend.close()

    def close(self) -> None:
        with self.condition:
            self.pending = None
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            self.error = 'Speech shutdown timed out; active utterance may finish'
