"""Tunable parameters of the hazard geometry, in one place.

Lengths are metres. They assume the cloud has at least a rough metric scale
(the pipeline always provides one, even if only from the phone-height prior).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HazardParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # -- preprocessing
    voxel: float | None = None  # None: choose from the cloud's height
    coarse_factor: float = 2.5  # normals are fitted on a cloud this many times coarser
    normal_radius_coarse: float = 3.2  # normal fitting radius, in coarse voxels
    outlier_std_ratio: float = 2.5

    # -- face segmentation
    steep_deg: float = 25.0  # ground steeper than this may belong to a face
    cluster_eps_spacings: float = 3.0  # neighbour radius for connectivity, in point spacings
    azimuth_tolerance_deg: float = 40.0
    min_face_height: float = 0.3
    min_face_length: float = 0.4
    min_face_points: int = 150

    # -- profiles
    slice_width: float | None = None  # None: from face length
    bin_height_voxels: float = 2.0
    column_voxels: float = 2.0  # width of the thin columns overhang is measured in
    column_spacings: float = 2.5  # ...but never narrower than this many point spacings
    group_gap: float = 0.2  # a gap this wide along the profile ends the surface
    front_reach: float = 1.0  # how far in front of the face to look for the floor

    # -- undercut and overhang
    undercut_method: str = "profile"  # "profile" | "normals"
    min_depth: float = 0.08  # smallest overhang worth reporting
    min_region_width: float = 0.2
    noise_sigmas: float = 4.0  # a slice must stand this far above the face's noise floor
    lip_gap_bins: int = 2
    min_hollow_bins: float = 3.0  # hollow must be this many height bins tall...
    min_hollow_height: float = 0.06  # ...and at least this tall in metres
    ray_tolerance: float = 0.25  # slack in the 'seen under the lip' test, as a share of hollow height
    undercut_max_lip_fraction: float = 0.6  # lip above this share of the face is an overhang
    downward_deg: float = 105.0  # normals method: slope beyond this faces down

    # -- water at the toe
    water_cell_voxels: float = 2.5
    water_image_sigmas: float = 3.0  # how unlike dry floor a gap must look
    water_min_area: float = 0.06  # square metres
    water_min_cells: int = 12
    water_min_surround: float = 0.6  # share of the floor around a patch that must be reconstructed

    # -- crest cracks
    crack_cell_voxels: float = 1.0
    crack_image_sigmas: float = 3.5
    crack_min_length: float = 0.4
    crack_max_thickness: float = 0.10
    crest_min_coverage: float = 0.5  # share of the strip behind the edge that must be reconstructed
