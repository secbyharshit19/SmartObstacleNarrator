"""Pure decision pipeline: no camera, GPU, or audio dependencies."""
import math
from .types import Detection, WarningEvent
from .geometry import DistanceEstimator, observe
from .temporal import Associator, Track
from .narration import eligible, score, sentence

class Pipeline:
    def __init__(self, cfg: dict, calibration: dict, camera_id: str = "0"):
        self.cfg = cfg
        self.distance = DistanceEstimator(calibration, camera_id)
        self.associator = Associator(cfg, calibration.get("relative_uncertainty", 0.25))
        self.sequence = 0
        self.latest_time = -float("inf")

    def reset(self) -> None:
        self.associator.reset()

    def process(self, detections: list[Detection], now: float, size: tuple[int,int]) -> tuple[list[Track], WarningEvent | None]:
        if now < self.latest_time:
            raise ValueError("Timestamps must be monotonic")
        self.latest_time = now
        w, h = size
        if w <= 0 or h <= 0:
            raise ValueError("Invalid frame size")
        accepted = []
        for d in detections:
            if d.label not in self.cfg["weights"] or not self.cfg["confidence"] <= d.confidence <= 1:
                continue
            if len(d.box) != 4 or not all(math.isfinite(v) for v in d.box):
                continue
            x1,y1,x2,y2 = d.box
            if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                continue
            accepted.append(observe(d, size, self.cfg, self.distance))
        tracks = self.associator.update(accepted, now, size)
        candidates = [t for t in tracks if eligible(t, now, self.cfg)]
        if not candidates:
            return tracks, None
        t = max(candidates, key=lambda x: score(x, now, self.cfg))
        self.sequence += 1
        event = WarningEvent(str(self.sequence), t.id, sentence(t, self.cfg.get("numeric_distance", False)),
                             score(t, now, self.cfg), now, now+self.cfg["warning_ttl"], t.signature)
        return tracks, event

    def acknowledge(self, event: WarningEvent, started: float) -> None:
        t = self.associator.tracks.get(event.track_id)
        if t:
            t.last_spoken, t.spoken_signature = started, event.signature

    def repeat(self, now: float) -> WarningEvent | None:
        tracks = [t for t in self.associator.tracks.values()
                  if now-t.last_seen < self.cfg["continuity_gap"]]
        if not tracks:
            return None
        t = max(tracks, key=lambda x: score(x, now, self.cfg))
        self.sequence += 1
        return WarningEvent(str(self.sequence), t.id, sentence(t), score(t, now, self.cfg),
                            now, now+self.cfg["warning_ttl"], t.signature)
