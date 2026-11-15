"""One-page inspection report as a PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from brink.domain import Hazard, Report, Severity
from brink.outputs.colors import HAZARD_NAME, HAZARD_RGB

_SEVERITY_TEXT = {Severity.DANGER: "DANGER", Severity.WARNING: "WARNING", Severity.NOTE: "NOTE"}
_RANK = {Severity.DANGER: 2, Severity.WARNING: 1, Severity.NOTE: 0}


def _measurement_text(hazard: Hazard) -> str:
    parts = []
    for m in hazard.measurements:
        if m.name in ("limit", "lip_height", "width"):
            continue
        unit = {"deg": " deg", "m": " m", "model": " units"}.get(m.unit, f" {m.unit}")
        bound = "at least " if m.is_lower_bound else ""
        error = f" (+/- {m.uncertainty:.2f})" if m.uncertainty is not None and m.unit != "deg" else ""
        error = f" (+/- {m.uncertainty:.0f})" if m.uncertainty is not None and m.unit == "deg" else error
        value = f"{m.value:.0f}" if m.unit == "deg" else f"{m.value:.2f}"
        parts.append(f"{m.name}: {bound}{value}{unit}{error}")
    return "; ".join(parts)


def save_pdf(report: Report, hazard_image: Path | None, path: str | Path) -> None:
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["BodyText"], fontSize=8, leading=10)
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8.5, leading=10.5)
    doc = SimpleDocTemplate(
        str(path), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
        title="Brink inspection record", author="Brink",
    )
    recon, capture = report.reconstruction, report.capture
    story = [Paragraph("Brink inspection record", styles["Title"])]

    when = capture.captured_at.strftime("%Y-%m-%d %H:%M %Z") if capture.captured_at else "not recorded in the video"
    gravity = f"{recon.gravity.source}, +/- {recon.gravity.uncertainty_deg:.1f} deg" if recon.gravity else "n/a"
    if recon.scale.is_metric:
        scale = f"{recon.scale.source.replace('_', ' ')}, +/- {100 * (recon.scale.relative_uncertainty or 0):.0f}%"
    else:
        scale = "none: lengths are not in metres"
    location = "not recorded"
    if report.site.latitude is not None and report.site.longitude is not None:
        location = f"{report.site.latitude:.5f}, {report.site.longitude:.5f}"
    meta = [
        ["Site", report.site.name or report.site.id, "Material stated", report.site.material.replace("_", " ")],
        ["Video", Path(capture.video_path).name, "Filmed", when],
        ["Location", location, "Report made", report.generated_at.strftime("%Y-%m-%d %H:%M UTC")],
        ["Frames used", f"{recon.frames_registered} of {recon.frames_selected}", "Direction of 'up' from", gravity],
        ["Size (scale) from", scale, "Video fingerprint", (capture.video_sha256 or "")[:24]],
    ]
    table = Table(meta, colWidths=[27 * mm, 62 * mm, 34 * mm, 59 * mm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.grey), ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    story += [table, Spacer(1, 4 * mm)]

    if not recon.success:
        story.append(Paragraph(
            f"<b>No result.</b> {recon.failure_reason or 'The video could not be turned into a 3D model.'} "
            "Nothing was checked, so this record says nothing about the ground.", styles["BodyText"],
        ))
    else:
        hazards = sorted(report.hazards, key=lambda h: (-_RANK[h.severity], -h.confidence))
        if hazards:
            story.append(Paragraph(f"<b>{len(hazards)} hazard(s) measured</b>", styles["Heading3"]))
            rows = [["", "Level", "What was measured", "Measurement", "Confidence"]]
            style = [
                ("FONTSIZE", (0, 0), (-1, 0), 8), ("TEXTCOLOR", (0, 0), (-1, 0), colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.lightgrey),
            ]
            for i, hazard in enumerate(hazards, start=1):
                rgb = HAZARD_RGB[hazard.type]
                rows.append([
                    "", Paragraph(f"<b>{_SEVERITY_TEXT[hazard.severity]}</b><br/>{HAZARD_NAME[hazard.type]}", cell),
                    Paragraph(hazard.message, cell), Paragraph(_measurement_text(hazard), cell),
                    Paragraph(f"{hazard.confidence_level.value} ({hazard.confidence:.2f})", cell),
                ])
                style.append(("BACKGROUND", (0, i), (0, i), colors.Color(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)))
            table = Table(rows, colWidths=[4 * mm, 30 * mm, 78 * mm, 46 * mm, 24 * mm], repeatRows=1)
            table.setStyle(TableStyle(style))
            story.append(table)
        else:
            story.append(Paragraph(
                "<b>No hazard was measured in this capture.</b> That is not a clearance: Brink only sees what the "
                "camera saw, and only checks for the shapes listed below.", styles["BodyText"],
            ))
        if hazard_image is not None and Path(hazard_image).exists():
            story += [Spacer(1, 3 * mm), Image(str(hazard_image), width=150 * mm, height=150 * mm * 9 / 16, kind="proportional")]

    story += [Spacer(1, 3 * mm)]
    status = "provisional, not approved by an engineer" if report.rules_provisional else "approved"
    story.append(Paragraph(
        "Checked for: undercut bases, overhangs, faces steeper than the stated material allows, cracks along "
        f"the top edge, standing water at the foot. Hazard rules version {report.rules_version} ({status}).", small,
    ))
    story.append(Spacer(1, 1.5 * mm))
    for note in report.limitations:
        story.append(Paragraph(f"&bull; {note}", small))
    doc.build(story)
