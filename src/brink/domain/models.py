"""Brink domain model.

Conventions used everywhere:
- World frame after orientation is right-handed with +Z up (against gravity).
- Lengths are metres when ``ScaleEstimate.is_metric`` is true. Otherwise they
  are in arbitrary model units and any length-based output must say so.
- Angles are degrees. A slope of 0 is flat ground, 90 is vertical, and more
  than 90 leans out over the open side (overhanging).
- Nothing here ever describes ground as free of risk. A report lists measured
  hazards; an empty list means "nothing was measured", never "nothing is there".
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

Vec3 = tuple[float, float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class HazardType(StrEnum):
    UNDERCUT = "undercut"
    OVERHANG = "overhang"
    OVERSTEEP = "oversteep"
    CREST_CRACK = "crest_crack"
    WATER_AT_TOE = "water_at_toe"


class Severity(StrEnum):
    """Outcome of applying the rules file to a measurement."""

    NOTE = "note"
    WARNING = "warning"
    DANGER = "danger"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_score(cls, score: float) -> ConfidenceLevel:
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.45:
            return cls.MEDIUM
        return cls.LOW


class Site(_Model):
    id: str
    name: str = ""
    material: str = "unknown"
    latitude: float | None = None
    longitude: float | None = None


class Capture(_Model):
    """One video of one site at one time."""

    id: str
    site_id: str = "unknown"
    video_path: str
    video_sha256: str | None = None
    captured_at: datetime | None = None
    device: str | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sensor_sidecar: str | None = None
    notes: str = ""


class GravityEstimate(_Model):
    """Direction of 'up' in the reconstruction frame, before reorientation."""

    up: Vec3
    source: str  # "sensor" | "scene" | "ground_truth"
    uncertainty_deg: float


class ScaleEstimate(_Model):
    """Multiply model units by ``factor`` to get metres."""

    factor: float = 1.0
    source: str = "none"  # "none" | "known_distance" | "camera_height" | "sensor" | ...
    relative_uncertainty: float | None = None

    @property
    def is_metric(self) -> bool:
        return self.source != "none"


class Reconstruction(_Model):
    capture_id: str
    frames_selected: int = 0
    frames_registered: int = 0
    points: int = 0
    mean_reprojection_error_px: float | None = None
    gravity: GravityEstimate | None = None
    scale: ScaleEstimate = Field(default_factory=ScaleEstimate)
    cloud_path: str | None = None
    success: bool = False
    failure_reason: str | None = None
    timings_s: dict[str, float] = Field(default_factory=dict)
    methods: dict[str, str] = Field(default_factory=dict)


class Measurement(_Model):
    name: str
    value: float
    unit: str
    uncertainty: float | None = None
    # True when the quantity could only be bounded from below, for example an
    # undercut whose back wall was hidden from every camera.
    is_lower_bound: bool = False


class Evidence(_Model):
    kind: str  # "geometry" | "image" | "photo_consistency" | ...
    detail: str = ""
    score: float | None = None


class Face(_Model):
    """A steep, roughly planar piece of ground: a pit wall, bench or pile side."""

    id: str
    label: str = ""
    centroid: Vec3
    # Unit horizontal vector pointing out of the ground into open air.
    outward: Vec3
    # Unit horizontal vector along the face.
    strike: Vec3
    slope_deg: float
    slope_uncertainty_deg: float
    steepest_slope_deg: float | None = None
    height: float
    height_uncertainty: float | None = None
    length: float
    toe_z: float
    crest_z: float
    n_points: int = 0
    coverage: float | None = None


class Hazard(_Model):
    id: str
    type: HazardType
    face_id: str | None = None
    location: Vec3
    # Extent along the face (strike direction), in model length units.
    extent_along_face: float | None = None
    measurements: list[Measurement] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    severity: Severity = Severity.NOTE
    rule_id: str | None = None
    message: str = ""
    evidence: list[Evidence] = Field(default_factory=list)

    @property
    def confidence_level(self) -> ConfidenceLevel:
        return ConfidenceLevel.from_score(self.confidence)

    def measurement(self, name: str) -> Measurement | None:
        return next((m for m in self.measurements if m.name == name), None)


class Report(_Model):
    schema_version: str = "0.1"
    generated_at: datetime
    site: Site
    capture: Capture
    reconstruction: Reconstruction
    faces: list[Face] = Field(default_factory=list)
    hazards: list[Hazard] = Field(default_factory=list)
    rules_version: str = ""
    rules_provisional: bool = True
    limitations: list[str] = Field(default_factory=list)
    artifact_sha256: dict[str, str] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)


# --- Phase 2 (movement between captures). Defined now so storage and APIs can
# --- plan for them; nothing in Phase 0 produces these.


class MovementRegion(_Model):
    id: str
    face_id: str | None = None
    location: Vec3
    area: float
    mean_displacement: float
    max_displacement: float
    level_of_detection: float
    confidence: float = Field(ge=0.0, le=1.0)


class Comparison(_Model):
    site_id: str
    reference_capture_id: str
    compared_capture_id: str
    alignment_rms: float | None = None
    regions: list[MovementRegion] = Field(default_factory=list)
