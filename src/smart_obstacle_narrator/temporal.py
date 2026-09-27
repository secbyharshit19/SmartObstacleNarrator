"""Small greedy one-to-one association; no identity or motion guarantee."""
from collections import deque
from dataclasses import dataclass, field, replace
from statistics import median
from .types import Observation
from .geometry import direction, band

def iou(a: tuple, b: tuple) -> float:
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union > 0 else 0.0

@dataclass
class Track:
    id: int
    observation: Observation
    first_seen: float
    last_seen: float
    count: int = 1
    distances: deque = field(default_factory=lambda: deque(maxlen=5))
    pending_direction: str | None = None
    direction_since: float = 0
    pending_band: str | None = None
    band_since: float = 0
    last_spoken: float = -float("inf")
    spoken_signature: tuple[str, str] | None = None

    @property
    def signature(self) -> tuple[str, str]:
        return self.observation.direction, self.observation.distance_band

class Associator:
    def __init__(self, cfg: dict, uncertainty: float):
        self.cfg, self.uncertainty = cfg, uncertainty
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, observations: list[Observation], now: float, size: tuple[int, int]) -> list[Track]:
        cfg = self.cfg
        self.tracks = {i:t for i,t in self.tracks.items() if now-t.last_seen <= cfg["disappear_seconds"]}
        pairs = []
        w, h = size
        for i,t in self.tracks.items():
            a = t.observation.detection
            for j,o in enumerate(observations):
                b = o.detection
                if a.label != b.label:
                    continue
                overlap = iou(a.box, b.box)
                dx = ((a.box[0]+a.box[2])-(b.box[0]+b.box[2]))/(2*w)
                dy = ((a.box[1]+a.box[3])-(b.box[1]+b.box[3]))/(2*h)
                distance = (dx*dx+dy*dy)**0.5
                if overlap >= 0.1 or distance <= 0.08:
                    pairs.append((overlap - distance, i, j))
        used_t, used_o, current = set(), set(), []
        for _,i,j in sorted(pairs, reverse=True):
            if i in used_t or j in used_o:
                continue
            used_t.add(i); used_o.add(j)
            t, o = self.tracks[i], observations[j]
            if now - t.last_seen > cfg["continuity_gap"]:
                t.first_seen, t.count = now, 0
                t.distances.clear()
                t.pending_band = t.pending_direction = None
            t.count += 1
            previous = t.observation
            proposed_direction = direction(o.detection.box, w, cfg, previous.direction)
            if proposed_direction == previous.direction:
                t.pending_direction = None
            elif proposed_direction != t.pending_direction:
                t.pending_direction, t.direction_since = proposed_direction, now
            stable_direction = (proposed_direction if t.pending_direction is not None and
                                now-t.direction_since >= cfg["state_hold"] else previous.direction)
            if o.distance_m is None:
                t.distances.clear()
                stable_band, filtered = "unknown", None
                t.pending_band = None
            else:
                t.distances.append(o.distance_m)
                filtered = median(t.distances)
                proposed_band = band(filtered, cfg, self.uncertainty, previous.distance_band)
                if proposed_band == previous.distance_band:
                    t.pending_band = None
                elif proposed_band != t.pending_band:
                    t.pending_band, t.band_since = proposed_band, now
                stable_band = (proposed_band if t.pending_band is not None and
                               now-t.band_since >= cfg["state_hold"] else previous.distance_band)
            t.observation = replace(o, direction=stable_direction, distance_m=filtered, distance_band=stable_band)
            t.last_seen = now
            current.append(t)
        for j,o in enumerate(observations):
            if j not in used_o:
                t = Track(self.next_id, o, now, now)
                if o.distance_m is not None:
                    t.distances.append(o.distance_m)
                self.tracks[t.id] = t
                self.next_id += 1
                current.append(t)
        return current
