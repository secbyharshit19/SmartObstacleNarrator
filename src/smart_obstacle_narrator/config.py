"""Configuration loading with fail-fast validation."""
from pathlib import Path
import math
import yaml
from .types import CLASSES

def read_yaml(path: str | Path) -> dict:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value

def load_config(path: str | Path) -> dict:
    cfg = read_yaml(path)
    required = ("confidence", "imgsz", "left_boundary", "right_boundary", "state_hold",
                "confirm_seconds", "confirm_observations", "continuity_gap", "disappear_seconds",
                "repeat_seconds", "speech_gap", "warning_ttl", "near_m", "close_m",
                "direction_hysteresis", "distance_hysteresis_m", "max_frame_age")
    for key in required:
        if key not in cfg or not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]):
            raise ValueError(f"Missing/invalid numeric setting: {key}")
        if cfg[key] < 0:
            raise ValueError(f"Setting must be nonnegative: {key}")
    if not 0 < cfg["confidence"] <= 1:
        raise ValueError("confidence must be in (0,1]")
    if not 0 < cfg["left_boundary"] < cfg["right_boundary"] < 1:
        raise ValueError("Invalid direction boundaries")
    if cfg["close_m"] >= cfg["near_m"] or cfg["confirm_observations"] < 1:
        raise ValueError("Invalid distance thresholds or confirmation count")
    if cfg["continuity_gap"] > cfg["disappear_seconds"]:
        raise ValueError("continuity_gap must not exceed disappear_seconds")
    if tuple(cfg.get("classes", ())) != CLASSES:
        raise ValueError("Class order must match the project contract")
    if set(cfg.get("weights", {})) != set(CLASSES):
        raise ValueError("A weight is required for every class")
    if any(not 0 <= float(v) <= 1 for v in cfg["weights"].values()):
        raise ValueError("Class weights must lie in [0,1]")
    return cfg
