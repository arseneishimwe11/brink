"""Tileable value noise, used for surface roughness and soil texture."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_LATTICE_SIZE = 64


def lattice(seed: int, size: int = _LATTICE_SIZE) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (size, size, size)).astype(np.float32)


def value_noise(points: NDArray, grid: NDArray[np.float32]) -> NDArray[np.float32]:
    """Smoothly interpolated lattice noise in [-1, 1] at ``points`` (..., 3)."""
    n = grid.shape[0]
    p = np.asarray(points, dtype=np.float32)
    base = np.floor(p)
    f = p - base
    i0 = base.astype(np.int64) % n
    i1 = (i0 + 1) % n
    u = f * f * (3.0 - 2.0 * f)
    x0, y0, z0 = i0[..., 0], i0[..., 1], i0[..., 2]
    x1, y1, z1 = i1[..., 0], i1[..., 1], i1[..., 2]
    ux, uy, uz = u[..., 0], u[..., 1], u[..., 2]
    c00 = grid[x0, y0, z0] * (1 - ux) + grid[x1, y0, z0] * ux
    c10 = grid[x0, y1, z0] * (1 - ux) + grid[x1, y1, z0] * ux
    c01 = grid[x0, y0, z1] * (1 - ux) + grid[x1, y0, z1] * ux
    c11 = grid[x0, y1, z1] * (1 - ux) + grid[x1, y1, z1] * ux
    c0 = c00 * (1 - uy) + c10 * uy
    c1 = c01 * (1 - uy) + c11 * uy
    return (c0 * (1 - uz) + c1 * uz).astype(np.float32)


def fbm(
    points: NDArray,
    grid: NDArray[np.float32],
    wavelength: float,
    octaves: int = 4,
    gain: float = 0.5,
    lacunarity: float = 2.03,
) -> NDArray[np.float32]:
    """Fractal sum of value noise, roughly in [-1, 1]."""
    p = np.asarray(points, dtype=np.float32) / np.float32(wavelength)
    total = np.zeros(p.shape[:-1], dtype=np.float32)
    amp, norm = 1.0, 0.0
    for k in range(octaves):
        total += np.float32(amp) * value_noise(p + np.float32(17.3 * k), grid)
        norm += amp
        amp *= gain
        p = p * np.float32(lacunarity)
    return total / np.float32(norm)
