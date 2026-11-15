"""Frame selection: keep sharp frames that overlap well, drop the rest.

A two-minute phone video has thousands of frames; camera solving and dense
matching need roughly a hundred. Frames are spaced by how far the image has
moved (not by time, so a pause adds nothing), and within each step the
sharpest candidate is kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict


class FrameSelectionParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_side: int = 960  # working resolution of the saved frames
    analysis_side: int = 480
    decode_fps: float = 10.0  # frames per second actually examined
    step_fraction: float = 0.06  # image motion between kept frames, as a share of width
    min_frames: int = 30
    max_frames: int = 80
    blur_reject_ratio: float = 0.35  # drop frames far blurrier than their neighbours


@dataclass
class SelectedFrames:
    directory: Path
    names: list[str]
    times: list[float]
    source_index: list[int]
    width: int
    height: int
    examined: int
    video_fps: float
    video_frames: int


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def _motion(prev: np.ndarray, cur: np.ndarray) -> float:
    """Median image motion in pixels between two small grey frames."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=300, qualityLevel=0.01, minDistance=8)
    if pts is None or len(pts) < 12:
        return float("nan")
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    ok = status.ravel() == 1
    if ok.sum() < 12:
        return float("nan")
    return float(np.median(np.linalg.norm((nxt - pts).reshape(-1, 2)[ok], axis=1)))


def select_frames(video: str | Path, out_dir: str | Path, params: FrameSelectionParams | None = None) -> SelectedFrames:
    params = params or FrameSelectionParams()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / params.decode_fps)))

    # Pass 1: sharpness and cumulative image motion of every examined frame.
    index, sharp, travel = [], [], []
    prev, moved, last_step, i = None, 0.0, 0.0, -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = params.analysis_side / max(h, w)
        small = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            step = _motion(prev, small)
            last_step = step if np.isfinite(step) else last_step
            moved += last_step / small.shape[1]
        index.append(i)
        sharp.append(_sharpness(small))
        travel.append(moved)
        prev = small
    cap.release()
    if len(index) < 3:
        raise RuntimeError("video has too few readable frames")
    sharp_a, travel_a = np.array(sharp), np.array(travel)

    # Pass 2: one frame per step of image motion, the sharpest near each mark.
    n = int(np.clip(round(travel_a[-1] / params.step_fraction), params.min_frames, params.max_frames))
    n = min(n, len(index))
    marks = np.linspace(0, travel_a[-1], n)
    half = 0.3 * (marks[1] - marks[0]) if n > 1 else 0.0
    chosen: list[int] = []
    for mark in marks:
        window = np.flatnonzero(np.abs(travel_a - mark) <= half)
        if len(window) == 0:
            window = np.array([int(np.argmin(np.abs(travel_a - mark)))])
        best = int(window[np.argmax(sharp_a[window])])
        if not chosen or best > chosen[-1]:
            chosen.append(best)
    # Drop frames that are much blurrier than the frames around them.
    local = np.array([np.median(sharp_a[chosen][max(0, k - 3) : k + 4]) for k in range(len(chosen))])
    chosen = [c for c, ref in zip(chosen, local, strict=True) if sharp_a[c] >= params.blur_reject_ratio * ref]

    # Pass 3: decode the chosen frames at working resolution.
    wanted = {index[c]: k for k, c in enumerate(chosen)}
    cap = cv2.VideoCapture(str(video))
    names, times, source = [], [], []
    width = height = 0
    i = -1
    last = max(wanted)
    while i < last:
        if not cap.grab():
            break
        i += 1
        if i not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        h, w = frame.shape[:2]
        scale = min(1.0, params.max_side / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        height, width = frame.shape[:2]
        name = f"frame_{wanted[i]:04d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        names.append(name)
        times.append(i / fps)
        source.append(i)
    cap.release()
    result = SelectedFrames(out_dir, names, times, source, width, height, len(index), fps, total)
    (out_dir / "frames.json").write_text(json.dumps({
        "names": names, "times": times, "source_index": source, "width": width, "height": height,
        "examined": len(index), "video_fps": fps, "video_frames": total,
    }))
    return result


def load_selected(out_dir: str | Path) -> SelectedFrames:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "frames.json").read_text())
    return SelectedFrames(out_dir, **meta)
