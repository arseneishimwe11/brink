"""Colours used consistently across the 3D model, the hazard image and the PDF."""

from __future__ import annotations

from brink.domain import HazardType

HAZARD_RGB: dict[HazardType, tuple[int, int, int]] = {
    HazardType.UNDERCUT: (220, 30, 30),
    HazardType.OVERHANG: (200, 30, 120),
    HazardType.OVERSTEEP: (240, 150, 20),
    HazardType.CREST_CRACK: (150, 50, 220),
    HazardType.WATER_AT_TOE: (30, 110, 230),
}

HAZARD_NAME: dict[HazardType, str] = {
    HazardType.UNDERCUT: "Undercut",
    HazardType.OVERHANG: "Overhang",
    HazardType.OVERSTEEP: "Over-steep face",
    HazardType.CREST_CRACK: "Crack at the top edge",
    HazardType.WATER_AT_TOE: "Water at the foot",
}
