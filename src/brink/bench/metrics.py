"""Scoring: match what Brink reported against what the scene really contains."""

from __future__ import annotations

from typing import Any

import numpy as np

from brink.domain import Face, Hazard, HazardType, Severity
from brink.rules.engine import Rules

OVERHANG_FAMILY = {HazardType.UNDERCUT, HazardType.OVERHANG}
_SEVERITY_RANK = {Severity.NOTE: 0, Severity.WARNING: 1, Severity.DANGER: 2}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def match_main_face(faces: list[Face], gt_face: dict) -> Face | None:
    """The reported face that looks the same way as the true test face."""
    gt_out = np.asarray(gt_face["outward"])
    best, best_points = None, 0
    for face in faces:
        if np.dot(face.outward, gt_out) > np.cos(np.radians(30)) and face.n_points > best_points:
            best, best_points = face, face.n_points
    return best


def _overlaps(hazard: Hazard, gt: dict, strike: np.ndarray, slack: float) -> bool:
    centre = float(np.dot(hazard.location, strike))
    gt_centre = float(np.dot(gt["location"], strike))
    half = 0.5 * (hazard.extent_along_face or 0.0) + 0.5 * gt["extent_along_face"] + slack
    return abs(centre - gt_centre) <= half


def score_case(
    faces: list[Face],
    hazards: list[Hazard],
    truth: dict,
    rules: Rules,
    material: str,
    to_truth: tuple[np.ndarray, float, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Compare one analysis with ground truth.

    ``to_truth`` is an optional (rotation, scale, shift) taking reported
    coordinates into the truth frame; reported lengths are multiplied by the
    scale only for location matching, never for the measurements themselves.
    """
    def to_gt(p):
        v = np.asarray(p, dtype=float)
        if to_truth is None:
            return v
        r, s, t = to_truth
        return s * (r @ v) + t

    gt_face = truth["faces"][0]
    strike = np.array([1.0, 0.0, 0.0])
    out: dict[str, Any] = {"n_faces": len(faces), "n_hazards": len(hazards)}

    main = match_main_face(
        [f.model_copy(update={"outward": tuple((to_truth[0] @ np.asarray(f.outward)) if to_truth else f.outward)}) for f in faces],
        gt_face,
    )
    out["face_found"] = main is not None
    if main is not None:
        out["slope_deg"] = main.slope_deg
        out["slope_err_deg"] = main.slope_deg - gt_face["slope_deg"]
        out["slope_unc_deg"] = main.slope_uncertainty_deg
        out["height"] = main.height
        out["height_err_rel"] = main.height / gt_face["height"] - 1.0

    # -- undercut / overhang
    gt_over = [h for h in truth["hazards"] if h["type"] in ("undercut", "overhang")]
    pred_over = [h for h in hazards if h.type in OVERHANG_FAMILY]
    located = [h.model_copy(update={"location": tuple(to_gt(h.location))}) for h in pred_over]
    slack = 0.25
    matched: set[int] = set()
    out["gt_overhangs"] = []
    for gt in gt_over:
        hits = [i for i, h in enumerate(located) if _overlaps(h, gt, strike, slack)]
        record: dict[str, Any] = {"type": gt["type"], "depth": gt["depth"], "detected": bool(hits)}
        if hits:
            best = max(hits, key=lambda i: located[i].measurement("depth").value)
            h = pred_over[best]
            d = h.measurement("depth")
            record.update(
                pred_type=h.type.value, pred_depth=d.value, depth_err=d.value - gt["depth"],
                lower_bound=d.is_lower_bound, confidence=h.confidence, severity=h.severity.value,
            )
            matched.update(hits)
        out["gt_overhangs"].append(record)
    spurious = [pred_over[i] for i in range(len(pred_over)) if i not in matched]
    out["false_overhangs"] = len(spurious)
    out["false_overhangs_warning"] = sum(_SEVERITY_RANK[h.severity] >= 1 for h in spurious)
    out["false_overhang_depths"] = [h.measurement("depth").value for h in spurious]
    out["false_overhang_conf"] = [h.confidence for h in spurious]

    # -- over-steep, judged only where the truth is clearly one side of the rule
    limit = rules.material(material).max_slope_deg + rules.defaults.slope_margin_deg
    gap = gt_face["slope_deg"] - limit
    expected = None if abs(gap) < 5 or rules.material(material).max_slope_deg >= 90 else gap > 0
    predicted = main is not None and any(
        h.type is HazardType.OVERSTEEP and h.face_id == main.id for h in hazards
    )
    out["oversteep_expected"] = expected
    out["oversteep_predicted"] = bool(predicted)
    return out


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse per-case records (one condition, one method) into headline numbers."""
    ok = [r for r in records if r.get("reconstructed", True)]
    summary: dict[str, Any] = {"captures": len(records), "reconstructed": len(ok)}

    gts = [(r, g) for r in ok for g in r["gt_overhangs"] if g["type"] == "undercut"]
    # Captures that failed to reconstruct count as misses.
    missed_by_failure = [
        g for r in records if not r.get("reconstructed", True) for g in r.get("gt_overhangs", []) if g["type"] == "undercut"
    ]

    def rate(lo: float, hi: float) -> dict[str, Any]:
        sel = [g for _, g in gts if lo <= g["depth"] < hi]
        failed = [g for g in missed_by_failure if lo <= g["depth"] < hi]
        n = len(sel) + len(failed)
        k = sum(g["detected"] for g in sel)
        return {"detected": k, "n": n, "rate": k / n if n else float("nan"), "ci95": wilson(k, n)}

    summary["undercut_ge_20cm"] = rate(0.195, 99)
    summary["undercut_lt_20cm"] = rate(0.0, 0.195)
    summary["undercut_20_30cm"] = rate(0.195, 0.30)
    summary["undercut_30_50cm"] = rate(0.30, 0.50)
    summary["undercut_ge_50cm"] = rate(0.50, 99)

    errs = [g["depth_err"] for _, g in gts if g["detected"] and g["depth"] >= 0.195]
    summary["depth_err_median_abs"] = _pct([abs(e) for e in errs], 50)
    summary["depth_err_p90_abs"] = _pct([abs(e) for e in errs], 90)
    summary["depth_err_bias"] = float(np.mean(errs)) if errs else float("nan")
    rel = [g["depth_err"] / g["depth"] for _, g in gts if g["detected"] and g["depth"] >= 0.195]
    summary["depth_err_rel_median_abs"] = _pct([abs(e) for e in rel], 50)

    over = [g for r in ok for g in r["gt_overhangs"] if g["type"] == "overhang"]
    summary["overhang"] = {"detected": sum(g["detected"] for g in over), "n": len(over)}

    controls = [r for r in ok if not r["gt_overhangs"]]
    summary["control_captures"] = len(controls)
    summary["control_with_false_alarm"] = sum(r["false_overhangs"] > 0 for r in controls)
    summary["control_with_false_warning"] = sum(r["false_overhangs_warning"] > 0 for r in controls)
    summary["false_alarms_per_capture"] = float(np.mean([r["false_overhangs"] for r in ok])) if ok else float("nan")
    summary["false_warnings_per_capture"] = float(np.mean([r["false_overhangs_warning"] for r in ok])) if ok else float("nan")

    slope = [r["slope_err_deg"] for r in ok if r.get("face_found")]
    summary["face_found"] = sum(bool(r.get("face_found")) for r in ok)
    summary["slope_err_median_abs"] = _pct([abs(e) for e in slope], 50)
    summary["slope_err_p90_abs"] = _pct([abs(e) for e in slope], 90)
    summary["slope_err_max_abs"] = max((abs(e) for e in slope), default=float("nan"))
    summary["slope_within_5deg"] = sum(abs(e) <= 5 for e in slope)
    summary["slope_n"] = len(slope)
    height = [r["height_err_rel"] for r in ok if r.get("face_found")]
    summary["height_err_rel_median_abs"] = _pct([abs(e) for e in height], 50)
    summary["height_err_rel_p90_abs"] = _pct([abs(e) for e in height], 90)

    judged = [r for r in ok if r["oversteep_expected"] is not None]
    summary["oversteep_correct"] = sum(r["oversteep_expected"] == r["oversteep_predicted"] for r in judged)
    summary["oversteep_n"] = len(judged)
    times = [r["time_s"] for r in records if "time_s" in r]
    summary["time_median_s"] = _pct(times, 50)
    summary["time_p90_s"] = _pct(times, 90)
    return summary
