"""Camera-relative direction and conservative known-height depth estimation."""
import math
from .types import Detection, Observation

def direction(box: tuple, width: int, cfg: dict, previous: str | None = None) -> str:
    if width <= 0:
        raise ValueError("Frame width must be positive")
    u = (box[0] + box[2]) / (2 * width)
    left, right, margin = cfg["left_boundary"], cfg["right_boundary"], cfg["direction_hysteresis"]
    if previous == "left" and u < left + margin:
        return "left"
    if previous == "right" and u > right - margin:
        return "right"
    if previous == "ahead" and left - margin <= u <= right + margin:
        return "ahead"
    return "left" if u < left else "right" if u > right else "ahead"

class DistanceEstimator:
    """Returns unknown unless camera and class-specific assumptions are configured."""
    def __init__(self, calibration: dict, camera_id: str):
        self.calibration = calibration
        self.camera_id = camera_id
        if calibration.get("enabled"):
            for key in ("width", "height", "focal_y_px", "valid_min_m", "valid_max_m"):
                v = calibration.get(key)
                if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise ValueError(f"Invalid calibration: {key}")
            if calibration["valid_min_m"] >= calibration["valid_max_m"]:
                raise ValueError("Invalid calibrated distance range")
            if not 0 <= calibration.get("relative_uncertainty", 0.25) < 1:
                raise ValueError("Invalid relative uncertainty")
            for h in calibration.get("object_heights_m", {}).values():
                if not isinstance(h, (int, float)) or not math.isfinite(h) or h <= 0:
                    raise ValueError("Object heights must be positive")

    def estimate(self, detection: Detection, size: tuple[int, int]) -> tuple[float | None, str]:
        c = self.calibration
        if not c.get("enabled"):
            return None, "uncalibrated"
        if str(c.get("camera_id")) != self.camera_id:
            return None, "camera mismatch"
        if size != (c["width"], c["height"]):
            return None, "resolution mismatch"
        height = c.get("object_heights_m", {}).get(detection.label)
        if height is None:
            return None, "unknown object height"
        x1, y1, x2, y2 = detection.box
        w, h = size
        if min(x1, y1) <= 2 or x2 >= w - 2 or y2 >= h - 2:
            return None, "clipped object"
        if y2 - y1 < 20:
            return None, "box too small"
        z = height * c["focal_y_px"] / (y2 - y1)
        if not c["valid_min_m"] <= z <= c["valid_max_m"]:
            return None, "outside calibrated range"
        return float(z), "known-height approximation"

def band(z: float | None, cfg: dict, uncertainty: float = 0, previous: str | None = None) -> str:
    if z is None:
        return "unknown"
    # Conservative band: use upper end of the configured uncertainty interval.
    value = z * (1 + uncertainty)
    close, near, margin = cfg["close_m"], cfg["near_m"], cfg["distance_hysteresis_m"]
    if previous == "close" and value <= close + margin:
        return "close"
    if previous == "near" and close - margin <= value <= near + margin:
        return "near"
    if previous == "far" and value >= near - margin:
        return "far"
    return "close" if value <= close else "near" if value <= near else "far"

def observe(d: Detection, size: tuple[int, int], cfg: dict, estimator: DistanceEstimator) -> Observation:
    w, _ = size
    z, reason = estimator.estimate(d, size)
    left, right = cfg["left_boundary"] * w, cfg["right_boundary"] * w
    overlap = max(0.0, min(d.box[2], right) - max(d.box[0], left)) / (right - left)
    return Observation(d, direction(d.box, w, cfg), z, reason,
                       band(z, cfg, estimator.calibration.get("relative_uncertainty", 0.25)), overlap)
