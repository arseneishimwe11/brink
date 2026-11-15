"""Write every output of one run into a folder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from brink.pipeline.runner import PipelineResult


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_outputs(result: PipelineResult, out_dir: str | Path) -> dict[str, Path]:
    """Model, hazard image, PDF, viewer and JSON summary. Returns their paths.

    The JSON is written last and lists a SHA-256 for each other file, so the
    set can later be checked for changes.
    """
    from brink.outputs.hazard_image import save_hazard_image
    from brink.outputs.model import save_model
    from brink.outputs.pdf import save_pdf
    from brink.outputs.viewer import save_viewer

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = result.report
    paths: dict[str, Path] = {}
    image_path = None
    if result.geometry is not None and result.sfm is not None and result.frames is not None:
        paths["model"] = out_dir / "model.ply"
        colours = save_model(result.geometry, paths["model"])
        image_path = out_dir / "hazards.jpg"
        used = save_hazard_image(
            result.geometry, colours, result.sfm, result.frames.directory, result.rotation, result.scale,
            report.reconstruction.scale.is_metric, image_path,
        )
        if used is None:
            image_path = None
        else:
            paths["hazard_image"] = image_path
        paths["viewer"] = save_viewer(result.geometry, colours, report, out_dir / "viewer")
    paths["pdf"] = out_dir / "report.pdf"
    save_pdf(report, image_path, paths["pdf"])
    report.artifact_sha256 = {
        name: _sha256(path) for name, path in paths.items() if name != "viewer"
    }
    paths["json"] = out_dir / "report.json"
    paths["json"].write_text(json.dumps(json.loads(report.model_dump_json()), indent=2))
    return paths
