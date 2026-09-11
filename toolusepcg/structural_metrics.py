"""Layout-level structural coherence metrics for tile maps."""

from __future__ import annotations

from collections import deque
from math import log

import numpy as np


def _clamped01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _low_is_good_score(value: float, good: float, bad: float) -> float:
    """Return 1 below ``good`` and 0 at or above ``bad``."""
    if bad <= good:
        return 1.0 if value <= good else 0.0
    return _clamped01(1.0 - (value - good) / (bad - good))


def _component_sizes(mask: np.ndarray) -> list[int]:
    """Return 4-connected component sizes for a boolean mask."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    sizes: list[int] = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            seen[y, x] = True
            q: deque[tuple[int, int]] = deque([(y, x)])
            size = 0
            while q:
                cy, cx = q.popleft()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            sizes.append(size)

    return sizes


def _compute_local_coherence(arr: np.ndarray) -> dict[str, float]:
    """Compute domain-agnostic local anti-fragmentation metrics."""
    h, w = arr.shape
    right_edges = h * max(0, w - 1)
    down_edges = max(0, h - 1) * w
    total_edges = right_edges + down_edges
    if total_edges:
        transitions = 0
        if w > 1:
            transitions += int(np.sum(arr[:, 1:] != arr[:, :-1]))
        if h > 1:
            transitions += int(np.sum(arr[1:, :] != arr[:-1, :]))
        transition_density = transitions / total_edges
    else:
        transition_density = 0.0

    same_neighbor = np.zeros(arr.shape, dtype=bool)
    if w > 1:
        same = arr[:, 1:] == arr[:, :-1]
        same_neighbor[:, 1:] |= same
        same_neighbor[:, :-1] |= same
    if h > 1:
        same = arr[1:, :] == arr[:-1, :]
        same_neighbor[1:, :] |= same
        same_neighbor[:-1, :] |= same
    singleton_ratio = float(np.sum(~same_neighbor)) / arr.size

    small_tiles = 0
    for tile in np.unique(arr):
        for size in _component_sizes(arr == tile):
            if size <= 2:
                small_tiles += size
    small_component_ratio = small_tiles / arr.size

    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    normalized_entropy = entropy / log(len(counts)) if len(counts) > 1 else 0.0

    transition_score = _low_is_good_score(transition_density, good=0.18, bad=0.55)
    singleton_score = _low_is_good_score(singleton_ratio, good=0.02, bad=0.20)
    component_score = _low_is_good_score(small_component_ratio, good=0.03, bad=0.18)
    local_score = 0.45 * transition_score + 0.30 * singleton_score + 0.25 * component_score

    return {
        "local_coherence_score": local_score,
        "transition_density": transition_density,
        "singleton_ratio": singleton_ratio,
        "small_component_ratio": small_component_ratio,
        "tile_entropy": normalized_entropy,
    }


def _range_score(value: float, low: float, high: float) -> float:
    """Reward values inside a band, with linear falloff outside it."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamped01(value / low) if low > 0 else 0.0
    return _clamped01((1.0 - value) / (1.0 - high)) if high < 1 else 0.0


def _band_score(value: float, good_low: float, good_high: float, bad_low: float, bad_high: float) -> float:
    """Reward a preferred band with configurable low/high failure points."""
    if good_low <= value <= good_high:
        return 1.0
    if value < good_low:
        if good_low <= bad_low:
            return 0.0
        return _clamped01((value - bad_low) / (good_low - bad_low))
    if bad_high <= good_high:
        return 0.0
    return _clamped01((bad_high - value) / (bad_high - good_high))


def _smb_ground_surface(arr: np.ndarray) -> list[int | None]:
    """Return the highest solid/support tile in the lower half of each column."""
    h, w = arr.shape
    support_tiles = {1, 2, 3, 4, 5}
    start_y = max(0, h // 2)
    surfaces: list[int | None] = []
    for x in range(w):
        solid_rows = [y for y in range(start_y, h) if int(arr[y, x]) in support_tiles]
        surfaces.append(min(solid_rows) if solid_rows else None)
    return surfaces


def _compute_smb_design_coherence(arr: np.ndarray) -> dict[str, float]:
    """Compute SMB-specific design coherence beyond local smoothness."""
    h, w = arr.shape
    local = _compute_local_coherence(arr)

    support_tiles = {1, 2, 3, 4, 5}
    enemy_tiles = {7, 8, 9}
    object_tiles = {3, 4, 5, 6, 7, 8, 9}

    surfaces = _smb_ground_surface(arr)
    ground_columns = [y for y in surfaces if y is not None]
    ground_coverage = len(ground_columns) / w if w else 0.0

    profile_pairs = [(a, b) for a, b in zip(surfaces, surfaces[1:]) if a is not None and b is not None]
    if profile_pairs:
        smooth_pairs = sum(1 for a, b in profile_pairs if abs(a - b) <= 2)
        ground_profile_score = smooth_pairs / len(profile_pairs)
    else:
        ground_profile_score = 0.0
    ground_presence_score = _range_score(ground_coverage, low=0.45, high=0.92)

    corridor_top = max(0, h // 4)
    corridor_bottom = max(corridor_top + 1, h - 2)
    corridor = arr[corridor_top:corridor_bottom, :]
    corridor_solid_ratio = float(np.isin(corridor, list(support_tiles)).sum()) / corridor.size
    corridor_open_score = _range_score(corridor_solid_ratio, low=0.06, high=0.36)

    enemy_positions = np.argwhere(np.isin(arr, list(enemy_tiles)))
    if len(enemy_positions) == 0:
        enemy_support_score = 0.65
    else:
        supported = 0
        for y, x in enemy_positions:
            if y + 1 < h and int(arr[y + 1, x]) in support_tiles:
                supported += 1
        enemy_support_score = supported / len(enemy_positions)

    object_ratio = float(np.isin(arr, list(object_tiles)).sum()) / arr.size
    object_density_score = _band_score(object_ratio, good_low=0.025, good_high=0.10, bad_low=0.0, bad_high=0.22)

    segments = np.array_split(arr, 4, axis=1) if w >= 4 else [arr]
    segment_counts = [int(np.isin(seg, list(object_tiles)).sum()) for seg in segments]
    active_segments = sum(1 for count in segment_counts if count > 0)
    object_pacing_score = active_segments / len(segments)
    if sum(segment_counts) > 0 and len(segment_counts) > 1:
        mean_count = sum(segment_counts) / len(segment_counts)
        imbalance = sum(abs(count - mean_count) for count in segment_counts)
        max_imbalance = 2 * sum(segment_counts) * (len(segment_counts) - 1) / len(segment_counts)
        object_pacing_score *= _clamped01(1.0 - imbalance / max_imbalance)

    local_score = local["local_coherence_score"]
    clutter_score = _low_is_good_score(local["transition_density"], good=0.12, bad=0.28)
    design_score = (
        0.18 * ground_profile_score
        + 0.18 * corridor_open_score
        + 0.20 * object_density_score
        + 0.10 * object_pacing_score
        + 0.20 * clutter_score
        + 0.14 * local_score
    )

    return {
        **local,
        "coherence_score": design_score,
        "design_coherence_score": design_score,
        "ground_presence_score": ground_presence_score,
        "ground_profile_score": ground_profile_score,
        "corridor_open_score": corridor_open_score,
        "enemy_support_score": enemy_support_score,
        "object_density_score": object_density_score,
        "object_pacing_score": object_pacing_score,
        "clutter_score": clutter_score,
        "ground_coverage": ground_coverage,
        "object_ratio": object_ratio,
    }


def compute_coherence_metrics(data: np.ndarray, problem_type: str | None = None) -> dict[str, float]:
    """Compute structural coherence metrics for a tile map."""
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.size == 0:
        return {
            "coherence_score": 0.0,
            "transition_density": 1.0,
            "singleton_ratio": 1.0,
            "small_component_ratio": 1.0,
            "tile_entropy": 0.0,
        }

    if problem_type in {"smb", "smb_static"}:
        return _compute_smb_design_coherence(arr)

    local = _compute_local_coherence(arr)
    return {
        **local,
        "coherence_score": local["local_coherence_score"],
    }
