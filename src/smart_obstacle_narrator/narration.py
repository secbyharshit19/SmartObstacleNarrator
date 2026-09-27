"""Explainable ranking and deterministic, uncertainty-aware wording."""
from .temporal import Track

def confirmed(t: Track, now: float, cfg: dict) -> bool:
    return t.count >= cfg["confirm_observations"] and now-t.first_seen >= cfg["confirm_seconds"]

def eligible(t: Track, now: float, cfg: dict) -> bool:
    if not confirmed(t, now, cfg):
        return False
    if t.spoken_signature is None or now-t.last_spoken >= cfg["repeat_seconds"]:
        return True
    old_direction, old_band = t.spoken_signature
    rank = {"unknown": 0, "far": 1, "near": 2, "close": 3}
    return t.observation.direction != old_direction or rank[t.observation.distance_band] > rank[old_band]

def score(t: Track, now: float, cfg: dict) -> float:
    o = t.observation
    proximity = {"unknown": 0.5, "far": 0.3, "near": 0.7, "close": 1.0}[o.distance_band]
    persistence = min(1.0, (now-t.first_seen)/max(0.001, cfg["confirm_seconds"]))
    return 100 * (0.35*cfg["weights"][o.detection.label] + 0.30*proximity +
                  0.20*o.central_overlap + 0.15*persistence) * (0.5+0.5*o.detection.confidence)

def sentence(t: Track, numeric: bool = False) -> str:
    o = t.observation
    place = "ahead" if o.direction == "ahead" else f"on your {o.direction}"
    prefix = o.detection.label.capitalize()
    if numeric and o.distance_m is not None:
        return f"{prefix} approximately {o.distance_m:.1f} meters {place}."
    proximity = {"close": " very close", "near": " nearby", "far": "", "unknown": ""}[o.distance_band]
    return f"{prefix}{proximity} {place}."
