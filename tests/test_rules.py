import re

from brink.domain import Face, Hazard, HazardType, Measurement
from brink.rules import load_rules
from brink.rules.engine import describe


def _face(slope: float, height: float = 2.0) -> Face:
    return Face(
        id="face-0", label="A", centroid=(0, 0, 1), outward=(0, -1, 0), strike=(1, 0, 0),
        slope_deg=slope, slope_uncertainty_deg=1.0, height=height, length=5.0, toe_z=0.0, crest_z=height,
    )


def test_default_rules_are_marked_provisional():
    rules = load_rules()
    assert rules.provisional is True
    assert rules.approved_by is None


def test_depth_thresholds_escalate():
    rules = load_rules()
    assert rules.judge_depth(HazardType.UNDERCUT, 0.05) is None
    assert rules.judge_depth(HazardType.UNDERCUT, 0.12).value == "note"
    assert rules.judge_depth(HazardType.UNDERCUT, 0.25).value == "warning"
    assert rules.judge_depth(HazardType.UNDERCUT, 0.80).value == "danger"


def test_slope_limit_depends_on_material():
    rules = load_rules()
    assert rules.judge_slope(_face(50), "dry_sand", metric=True) is not None
    assert rules.judge_slope(_face(50), "clay", metric=True) is None
    assert rules.judge_slope(_face(89), "hard_rock", metric=True) is None
    # Unknown material falls back to the most cautious class.
    assert rules.judge_slope(_face(50), "no_such_material", metric=True) is not None


def test_no_message_ever_says_safe():
    for hazard_type in HazardType:
        hazard = Hazard(
            id="hz", type=hazard_type, location=(0, 0, 0), confidence=0.8,
            measurements=[
                Measurement(name="depth", value=0.4, unit="m"),
                Measurement(name="slope", value=70, unit="deg"),
                Measurement(name="limit", value=45, unit="deg"),
                Measurement(name="length", value=1.2, unit="m"),
            ],
        )
        for metric in (True, False):
            message = describe(hazard, "B", metric)
            assert message
            assert not re.search(r"\bsafe", message, re.IGNORECASE)
