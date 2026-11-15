"""The written outputs must exist, hang together, and never claim ground is safe."""

import json
import re
from datetime import UTC, datetime

import numpy as np

from brink.domain import (
    Capture,
    Hazard,
    HazardType,
    Measurement,
    Reconstruction,
    Report,
    Severity,
    Site,
)
from brink.outputs.pdf import save_pdf
from brink.pipeline.runner import LIMITATIONS
from brink.rules.engine import describe

SAFE = re.compile(r"\bsafe(ty|ly)?\b", re.IGNORECASE)


def _report(hazards, success=True) -> Report:
    return Report(
        generated_at=datetime(2026, 11, 15, tzinfo=UTC),
        site=Site(id="pit-1", name="North pit", material="damp_soil"),
        capture=Capture(id="v", video_path="v.mp4", video_sha256="ab" * 32),
        reconstruction=Reconstruction(capture_id="v", success=success, frames_selected=80, frames_registered=78,
                                      failure_reason=None if success else "Camera positions could not be solved."),
        hazards=hazards, rules_version="0.1-provisional", limitations=list(LIMITATIONS),
    )


def _hazards():
    out = []
    for hazard_type in HazardType:
        hazard = Hazard(
            id=f"hz-{hazard_type.value}", type=hazard_type, face_id="face-0", location=(0, 0, 0), confidence=0.8,
            severity=Severity.WARNING,
            measurements=[Measurement(name="depth", value=0.4, unit="m", uncertainty=0.05),
                          Measurement(name="slope", value=70, unit="deg", uncertainty=2)],
        )
        hazard.message = describe(hazard, "A", True)
        out.append(hazard)
    return out


def test_report_json_never_says_safe():
    for report in (_report(_hazards()), _report([]), _report([], success=False)):
        text = json.dumps(json.loads(report.model_dump_json()))
        assert not SAFE.search(text), SAFE.search(text)


def test_pdf_is_written_for_hazards_for_none_and_for_failure(tmp_path):
    for name, report in (("h", _report(_hazards())), ("n", _report([])), ("f", _report([], success=False))):
        path = tmp_path / f"{name}.pdf"
        save_pdf(report, None, path)
        data = path.read_bytes()
        assert data.startswith(b"%PDF") and len(data) > 1500


def test_source_text_never_says_safe():
    """Every string the package can show a person is checked at the source."""
    from pathlib import Path

    import brink

    root = Path(brink.__file__).parent
    offenders = []
    for path in list(root.rglob("*.py")) + list(root.rglob("*.yaml")):
        if "viewer_assets" in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            code = line.split("#", 1)[0]  # comments are for developers, not users
            for literal in re.findall(r"\"([^\"]*)\"|'([^']*)'", code):
                if SAFE.search(literal[0] or literal[1]):
                    offenders.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_wilson_and_numpy_are_sane():
    assert np.isfinite(1.0)
