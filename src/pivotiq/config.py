"""PivotIQ runtime configuration.

All thresholds and tunable parameters live here so they can be overridden
from a TOML file without touching source code.

Usage::

    from pivotiq.config import PivotIQConfig
    cfg = PivotIQConfig.from_toml(Path("pivotiq.toml"))
    # or:
    cfg = PivotIQConfig.default()
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]  # backport for ≤3.10


@dataclass
class DetectionConfig:
    """YOLO detection thresholds and model paths."""

    player_model: str = "yolo11n.pt"
    ball_model: Optional[str] = None   # None → falls back to player_model + warning
    player_confidence: float = 0.40
    ball_confidence: float = 0.30
    device: str = "auto"               # "auto" | "cpu" | "cuda" | "mps"
    player_class_id: int = 0           # COCO person class


@dataclass
class TrackingConfig:
    """ByteTrack player tracking parameters."""

    track_thresh: float = 0.50
    track_buffer: int = 30             # frames to keep lost track alive
    match_thresh: float = 0.80
    min_box_area: float = 10.0         # px² — ignore tiny detections


@dataclass
class RegistrationConfig:
    """Dynamic field registration parameters."""

    min_visible_keypoints: int = 4     # RANSAC needs ≥4 point correspondences
    ransac_reproj_threshold: float = 4.0   # pixels
    temporal_smoothing_alpha: float = 0.70  # exponential smoothing for H propagation
    optical_flow_max_corners: int = 200
    optical_flow_quality: float = 0.01


@dataclass
class EventConfig:
    """Shot / goal / chance detection thresholds."""

    shot_speed_threshold_mps: float = 5.0   # min ball speed to flag as a shot
    chance_xg_threshold: float = 0.20       # xG above this → big chance
    buildup_seconds: float = 30.0           # pre-event clip length
    goal_rebound_frames: int = 10           # frames to look for rebound after goal line


@dataclass
class PivotIQConfig:
    """Top-level config object; one instance per analysis run."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    registration: RegistrationConfig = field(default_factory=RegistrationConfig)
    events: EventConfig = field(default_factory=EventConfig)

    @classmethod
    def default(cls) -> PivotIQConfig:
        return cls()

    @classmethod
    def from_toml(cls, path: Path) -> PivotIQConfig:
        """Load config from a TOML file; unset keys keep their defaults.

        Raises :class:`ConfigError` for a missing/unparsable file, unknown sections or keys
        (typos must not be silently ignored), wrong value types and out-of-range values.
        """
        try:
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {path}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: invalid TOML ({exc})") from exc

        cfg = cls()
        unknown = sorted(set(raw) - set(_SECTIONS))
        if unknown:
            raise ConfigError(f"{path}: unknown section(s) {unknown}; expected {sorted(_SECTIONS)}")
        for key, dcls in _SECTIONS.items():
            if key not in raw:
                continue
            values = raw[key]
            if not isinstance(values, dict):
                raise ConfigError(f"{path}: [{key}] must be a table")
            current = asdict(getattr(cfg, key))
            bad = sorted(set(values) - set(current))
            if bad:
                raise ConfigError(f"{path}: unknown key(s) {bad} in [{key}]; valid: {sorted(current)}")
            current.update(values)
            setattr(cfg, key, dcls(**current))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Check types and ranges of every field; raises :class:`ConfigError`."""
        for section, dcls in _SECTIONS.items():
            obj = getattr(self, section)
            for f in fields(dcls):
                value = getattr(obj, f.name)
                _check_type(f"{section}.{f.name}", value, f.type)
        d, t, r, e = self.detection, self.tracking, self.registration, self.events
        _check_range("detection.player_confidence", d.player_confidence, 0.0, 1.0)
        _check_range("detection.ball_confidence", d.ball_confidence, 0.0, 1.0)
        if d.device not in DEVICES:
            raise ConfigError(f"detection.device must be one of {DEVICES}, got {d.device!r}")
        _check_range("tracking.track_thresh", t.track_thresh, 0.0, 1.0)
        _check_range("tracking.match_thresh", t.match_thresh, 0.0, 1.0)
        if t.track_buffer < 1:
            raise ConfigError(f"tracking.track_buffer must be >= 1, got {t.track_buffer}")
        if t.min_box_area < 0:
            raise ConfigError(f"tracking.min_box_area must be >= 0, got {t.min_box_area}")
        if r.min_visible_keypoints < 4:
            raise ConfigError(
                f"registration.min_visible_keypoints must be >= 4 (homography needs 4 points), "
                f"got {r.min_visible_keypoints}"
            )
        if r.ransac_reproj_threshold <= 0:
            raise ConfigError("registration.ransac_reproj_threshold must be > 0")
        _check_range("registration.temporal_smoothing_alpha", r.temporal_smoothing_alpha, 0.0, 1.0)
        _check_range("registration.optical_flow_quality", r.optical_flow_quality, 0.0, 1.0)
        if r.optical_flow_max_corners < 1:
            raise ConfigError("registration.optical_flow_max_corners must be >= 1")
        if e.shot_speed_threshold_mps <= 0:
            raise ConfigError("events.shot_speed_threshold_mps must be > 0")
        _check_range("events.chance_xg_threshold", e.chance_xg_threshold, 0.0, 1.0)
        if e.buildup_seconds < 0:
            raise ConfigError("events.buildup_seconds must be >= 0")
        if e.goal_rebound_frames < 0:
            raise ConfigError("events.goal_rebound_frames must be >= 0")


DEVICES = ("auto", "cpu", "cuda", "mps")

_SECTIONS = {
    "detection": DetectionConfig,
    "tracking": TrackingConfig,
    "registration": RegistrationConfig,
    "events": EventConfig,
}


class ConfigError(ValueError):
    """The configuration is invalid; the message says which key and why."""


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not lo <= value <= hi:
        raise ConfigError(f"{name} must be between {lo} and {hi}, got {value}")


def _check_type(name: str, value: object, annotation: object) -> None:
    # `from __future__ import annotations` makes dataclass annotations strings.
    ann = str(annotation)
    if "Optional" in ann and value is None:
        return
    if value is None or isinstance(value, bool):
        raise ConfigError(f"{name} has the wrong type: {type(value).__name__}")
    if ann.startswith("int"):
        ok = isinstance(value, int)
    elif ann.startswith("float"):
        ok = isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
    elif "str" in ann:
        ok = isinstance(value, str)
    else:
        ok = True
    if not ok:
        raise ConfigError(f"{name} must be {ann.replace('Optional[', '').rstrip(']')}, got {value!r}")
