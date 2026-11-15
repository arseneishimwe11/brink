"""From an oriented, roughly metric point cloud to faces and geometric hazards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brink.domain import Evidence, Face, Hazard, HazardType, Measurement, Severity
from brink.geometry import PointCloud
from brink.hazards.faces import FaceSegment, segment_faces
from brink.hazards.params import HazardParams
from brink.hazards.preprocess import preprocess, slope_degrees
from brink.hazards.profiles import FaceProfiles, compute_profiles
from brink.hazards.undercut import OverhangRegion, make_detector
from brink.rules.engine import Rules, describe


@dataclass
class GeometryResult:
    cloud: PointCloud  # cleaned cloud with normals
    voxel: float
    faces: list[Face] = field(default_factory=list)
    hazards: list[Hazard] = field(default_factory=list)
    segments: list[FaceSegment] = field(default_factory=list)
    profiles: list[FaceProfiles] = field(default_factory=list)
    # Per point: index of the hazard it belongs to, or -1.
    hazard_of_point: np.ndarray | None = None


def _label(i: int) -> str:
    return chr(ord("A") + i) if i < 26 else f"F{i}"


def _robust(values: np.ndarray) -> tuple[float, float]:
    med = float(np.median(values))
    spread = float(1.4826 * np.median(np.abs(values - med)))
    return med, spread


def _summarise_face(i: int, seg: FaceSegment, prof: FaceProfiles, cloud: PointCloud, gravity_unc: float, slope_method: str) -> Face | None:
    slices = prof.slices
    if not slices:
        return None
    heights = np.array([p.height for p in slices])
    height = float(np.percentile(heights, 90))
    slopes = np.array([p.slope_deg for p in slices if np.isfinite(p.slope_deg) and p.height > 0.6 * height])
    if len(slopes) >= 3:
        slope, spread = _robust(slopes)
        stat_unc = spread / np.sqrt(len(slopes))
        steepest = float(np.percentile(slopes, 90))
    else:
        slope, stat_unc, steepest = prof.plane_slope_deg, 3.0, prof.plane_slope_deg
    if slope_method == "plane" and np.isfinite(prof.plane_slope_deg):
        slope = prof.plane_slope_deg
    if not np.isfinite(slope):
        return None
    tall = [p for p in slices if p.height > 0.5 * height]
    length = (tall[-1].s_center - tall[0].s_center + prof.slice_width) if tall else 0.0
    xyz = cloud.xyz[seg.index]
    h_spread = float(1.4826 * np.median(np.abs(heights[heights > 0.5 * height] - np.median(heights[heights > 0.5 * height]))))
    return Face(
        id=f"face-{i}",
        label=_label(i),
        centroid=tuple(float(v) for v in xyz.mean(0)),
        outward=tuple(float(v) for v in seg.outward),
        strike=tuple(float(v) for v in seg.s_dir),
        slope_deg=float(slope),
        slope_uncertainty_deg=float(np.hypot(np.hypot(stat_unc, gravity_unc), 1.0)),
        steepest_slope_deg=steepest,
        height=height,
        height_uncertainty=max(h_spread, 2 * prof.bin_height),
        length=float(length),
        toe_z=float(np.median([p.z_floor for p in tall])) if tall else float(xyz[:, 2].min()),
        crest_z=float(np.percentile([p.z_crest for p in slices], 90)),
        n_points=int(len(seg.index)),
    )


def _depth_confidence(region: OverhangRegion, noise: float, params: HazardParams) -> float:
    """Heuristic 0-1 score: how far above the noise, and how widely supported."""
    snr = region.depth / max(noise, 1e-3)
    strength = 1 - np.exp(-max(snr - 1.0, 0.0) / 2.0)
    support = min(1.0, region.support / 5.0)
    agreement = 1.0
    if np.isfinite(region.depth_spread) and region.depth > 0:
        agreement = float(np.clip(1 - region.depth_spread / region.depth, 0.3, 1.0))
    score = 0.15 + 0.85 * strength * (0.5 + 0.5 * support) * agreement
    if not region.back_seen:
        score *= 0.9
    return float(np.clip(score, 0.05, 0.99))


def analyze_geometry(
    cloud: PointCloud,
    rules: Rules,
    material: str = "unknown",
    params: HazardParams | None = None,
    metric: bool = True,
    gravity_uncertainty_deg: float = 0.0,
    slope_method: str = "profile",
) -> GeometryResult:
    """Measure faces and geometric hazards (undercut, overhang, over-steep).

    ``cloud`` must already be rotated so +Z is up and scaled to rough metres.
    """
    params = params or HazardParams()
    clean, voxel, spacing = preprocess(cloud, params)
    result = GeometryResult(clean, voxel)
    slope = slope_degrees(clean.normals)
    segments, face_of = segment_faces(clean, voxel, spacing, params)
    detector = make_detector(params.undercut_method)
    hazard_of_point = np.full(len(clean), -1, dtype=np.int32)
    xyz = clean.xyz.astype(np.float64)

    for face_index, seg in enumerate(segments):
        prof = compute_profiles(clean, slope, face_of, segments, face_index, voxel, spacing, params)
        face = _summarise_face(len(result.faces), seg, prof, clean, gravity_uncertainty_deg, slope_method)
        if face is None:
            continue
        result.faces.append(face)
        result.segments.append(seg)
        result.profiles.append(prof)

        # Typical overhang reading on stretches with no hazard: the noise floor.
        depths = np.array([p.undercut.depth for p in prof.slices])
        noise = max(float(np.percentile(depths, 20)), 0.0) + 2 * voxel
        s_all, t_all = xyz @ seg.s_dir, xyz @ seg.t_dir

        for region in detector.detect(clean, slope, prof, voxel, params):
            htype = HazardType.UNDERCUT if region.kind == "undercut" else HazardType.OVERHANG
            severity = rules.judge_depth(htype, region.depth) if metric else Severity.WARNING
            if severity is None:
                continue
            centre = region.s_center * seg.s_dir + region.t_center * seg.t_dir
            centre[2] = 0.5 * (region.lip_z + region.hollow_z)
            hazard = Hazard(
                id=f"hz-{len(result.hazards)}",
                type=htype,
                face_id=face.id,
                location=tuple(float(v) for v in centre),
                extent_along_face=region.width,
                measurements=[
                    Measurement(
                        name="depth", value=region.depth, unit="m" if metric else "model",
                        uncertainty=max(region.depth_spread if np.isfinite(region.depth_spread) else 0.0, noise),
                        is_lower_bound=not region.back_seen,
                    ),
                    Measurement(name="lip_height", value=region.lip_z - face.toe_z, unit="m" if metric else "model"),
                    Measurement(name="width", value=region.width, unit="m" if metric else "model"),
                ],
                confidence=_depth_confidence(region, noise, params),
                severity=severity,
                rule_id=f"{htype.value}_depth_m",
                evidence=[Evidence(kind="geometry", detail=f"{detector.name} method, {region.support} supporting samples")],
            )
            hazard.message = describe(hazard, face.label, metric)
            in_region = (
                (s_all >= region.s_min) & (s_all <= region.s_max)
                & (xyz[:, 2] <= region.lip_z + 2 * voxel) & (xyz[:, 2] >= region.hollow_z - 2 * voxel)
                & (np.abs(t_all - region.t_center) <= region.depth)
            )
            hazard_of_point[in_region] = len(result.hazards)
            result.hazards.append(hazard)

        lean_deg = face.slope_deg - 90.0
        if lean_deg > 0:
            # The whole face tips out over its toe.
            depth = face.height * float(np.tan(np.radians(lean_deg)))
            severity = rules.judge_depth(HazardType.OVERHANG, depth) if metric else Severity.WARNING
            if severity is not None:
                hazard = Hazard(
                    id=f"hz-{len(result.hazards)}",
                    type=HazardType.OVERHANG,
                    face_id=face.id,
                    location=face.centroid,
                    extent_along_face=face.length,
                    measurements=[
                        Measurement(
                            name="depth", value=depth, unit="m" if metric else "model",
                            uncertainty=face.height * float(np.tan(np.radians(face.slope_uncertainty_deg))),
                        ),
                        Measurement(name="slope", value=face.slope_deg, unit="deg", uncertainty=face.slope_uncertainty_deg),
                    ],
                    confidence=float(np.clip(0.5 + 0.5 * np.tanh(lean_deg / (2 * face.slope_uncertainty_deg)), 0.05, 0.99)),
                    severity=severity,
                    rule_id="overhang_depth_m",
                    evidence=[Evidence(kind="geometry", detail="face leans out past vertical")],
                )
                hazard.message = describe(hazard, face.label, metric)
                result.hazards.append(hazard)

        severity = rules.judge_slope(face, material, metric)
        if severity is not None:
            limit = rules.material(material).max_slope_deg
            excess = face.slope_deg - limit
            hazard = Hazard(
                id=f"hz-{len(result.hazards)}",
                type=HazardType.OVERSTEEP,
                face_id=face.id,
                location=face.centroid,
                extent_along_face=face.length,
                measurements=[
                    Measurement(name="slope", value=face.slope_deg, unit="deg", uncertainty=face.slope_uncertainty_deg),
                    Measurement(name="limit", value=limit, unit="deg"),
                    Measurement(name="height", value=face.height, unit="m" if metric else "model", uncertainty=face.height_uncertainty),
                ],
                confidence=float(np.clip(0.5 + 0.5 * np.tanh((excess - rules.defaults.slope_margin_deg) / (2 * face.slope_uncertainty_deg)), 0.05, 0.99)),
                severity=severity,
                rule_id=f"max_slope_deg:{material}",
                evidence=[Evidence(kind="geometry", detail=f"profile slope over {len(prof.slices)} cross-sections")],
            )
            hazard.message = describe(hazard, face.label, metric)
            result.hazards.append(hazard)

    result.hazard_of_point = hazard_of_point
    return result
