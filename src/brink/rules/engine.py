"""Turn measurements into severities and plain warnings.

Geometry code measures; this module judges. Keeping them apart means an
engineer can change every threshold by editing one YAML file.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from brink.domain import Face, Hazard, HazardType, Measurement, Severity

_ORDER = [Severity.NOTE, Severity.WARNING, Severity.DANGER]


class Thresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    note: float
    warning: float
    danger: float

    def severity(self, value: float) -> Severity | None:
        if value >= self.danger:
            return Severity.DANGER
        if value >= self.warning:
            return Severity.WARNING
        if value >= self.note:
            return Severity.NOTE
        return None


class FixedSeverity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    severity: Severity


class Defaults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    undercut_depth_m: Thresholds
    overhang_depth_m: Thresholds
    crest_crack: FixedSeverity
    water_at_toe: FixedSeverity
    high_face_m: float
    slope_margin_deg: float


class Material(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str = ""
    max_slope_deg: float


class Rules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    provisional: bool = True
    approved_by: str | None = None
    defaults: Defaults
    materials: dict[str, Material]

    def material(self, name: str) -> Material:
        return self.materials.get(name) or self.materials["unknown"]

    # -- judging ---------------------------------------------------------

    def judge_depth(self, hazard_type: HazardType, depth_m: float) -> Severity | None:
        table = (
            self.defaults.undercut_depth_m
            if hazard_type is HazardType.UNDERCUT
            else self.defaults.overhang_depth_m
        )
        return table.severity(depth_m)

    def judge_slope(self, face: Face, material: str, metric: bool) -> Severity | None:
        """Severity for a face steeper than its material allows, else None."""
        limit = self.material(material).max_slope_deg
        if limit >= 90:
            return None
        excess = face.slope_deg - (limit + self.defaults.slope_margin_deg)
        if excess < 0:
            return None
        severity = Severity.WARNING if excess < 15 else Severity.DANGER
        if metric and face.height >= self.defaults.high_face_m:
            severity = _raise(severity)
        return severity

    def fixed(self, hazard_type: HazardType) -> Severity:
        if hazard_type is HazardType.CREST_CRACK:
            return self.defaults.crest_crack.severity
        return self.defaults.water_at_toe.severity


def _raise(severity: Severity) -> Severity:
    return _ORDER[min(_ORDER.index(severity) + 1, len(_ORDER) - 1)]


def load_rules(path: str | Path | None = None) -> Rules:
    if path is None:
        text = resources.files("brink.rules").joinpath("default_rules.yaml").read_text()
    else:
        text = Path(path).read_text()
    return Rules.model_validate(yaml.safe_load(text))


# -- wording -------------------------------------------------------------


def _fmt_len(m: Measurement, metric: bool) -> str:
    unit = "m" if metric else "model units"
    bound = "at least " if m.is_lower_bound else ""
    return f"{bound}{m.value:.2f} {unit}"


def describe(hazard: Hazard, face_label: str, metric: bool) -> str:
    """One plain sentence stating what was measured and what to avoid."""
    where = f"Face {face_label}" if face_label else "A face"
    t = hazard.type
    if t is HazardType.UNDERCUT:
        d = hazard.measurement("depth")
        size = _fmt_len(d, metric) if d else "an unmeasured amount"
        return f"{where} is undercut by {size} at its base. Do not work below it."
    if t is HazardType.OVERHANG:
        d = hazard.measurement("depth")
        size = _fmt_len(d, metric) if d else "an unmeasured amount"
        return f"{where} overhangs by {size}. Do not work below it."
    if t is HazardType.OVERSTEEP:
        a = hazard.measurement("slope")
        lim = hazard.measurement("limit")
        angle = f"{a.value:.0f} degrees" if a else "steeper than its limit"
        limit = f" (provisional limit {lim.value:.0f})" if lim else ""
        return f"{where} stands at {angle}{limit}. Keep people and machines back from its foot."
    if t is HazardType.CREST_CRACK:
        length = hazard.measurement("length")
        size = f" about {_fmt_len(length, metric)} long" if length else ""
        return f"{where} has a crack{size} along its top edge. Stay off the edge and out from under it."
    if t is HazardType.WATER_AT_TOE:
        return f"Standing water at the foot of {where[0].lower() + where[1:]}. Water weakens the base."
    return f"{where}: {t.value}."
