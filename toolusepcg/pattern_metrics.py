"""Tile-pattern distribution and divergence metrics for level analysis."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np


Pattern = tuple[int, ...]
Distribution = dict[Pattern, float]


SMB_REFERENCE_CHAR_TILES = {
    "-": 0,
    "X": 1,
    "#": 2,
    "S": 3,
    "Q": 4,
    "t": 5,
    "o": 6,
    "g": 7,
    "k": 8,
    "r": 8,
    "y": 9,
}


def pattern_counts(data: np.ndarray, window: tuple[int, int] = (2, 2)) -> Counter[Pattern]:
    """Count overlapping tile patterns in a 2D level array."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError("pattern metrics require a 2D level array")

    wh, ww = window
    if wh <= 0 or ww <= 0:
        raise ValueError("pattern window dimensions must be positive")
    if arr.shape[0] < wh or arr.shape[1] < ww:
        return Counter()

    counts: Counter[Pattern] = Counter()
    for y in range(arr.shape[0] - wh + 1):
        for x in range(arr.shape[1] - ww + 1):
            counts[tuple(int(v) for v in arr[y : y + wh, x : x + ww].ravel())] += 1
    return counts


def corpus_pattern_counts(arrays: list[np.ndarray], window: tuple[int, int] = (2, 2)) -> Counter[Pattern]:
    """Count overlapping tile patterns over a corpus of level arrays."""
    counts: Counter[Pattern] = Counter()
    for data in arrays:
        counts.update(pattern_counts(data, window))
    return counts


def normalize_counts(counts: Counter[Pattern]) -> Distribution:
    """Convert pattern counts into a probability distribution."""
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {pattern: count / total for pattern, count in counts.items()}


def kl_divergence(p: Distribution, q: Distribution, epsilon: float = 1e-8) -> float:
    """Compute smoothed KL(P || Q) over the union of observed patterns."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    p_values = np.array([p.get(key, 0.0) + epsilon for key in keys], dtype=float)
    q_values = np.array([q.get(key, 0.0) + epsilon for key in keys], dtype=float)
    p_values /= p_values.sum()
    q_values /= q_values.sum()
    return float(np.sum(p_values * np.log(p_values / q_values)))


def js_divergence(p: Distribution, q: Distribution, epsilon: float = 1e-8) -> float:
    """Compute smoothed Jensen-Shannon divergence between two distributions."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0

    p_values = np.array([p.get(key, 0.0) + epsilon for key in keys], dtype=float)
    q_values = np.array([q.get(key, 0.0) + epsilon for key in keys], dtype=float)
    p_values /= p_values.sum()
    q_values /= q_values.sum()
    mixed = 0.5 * (p_values + q_values)
    return float(
        0.5 * np.sum(p_values * np.log(p_values / mixed))
        + 0.5 * np.sum(q_values * np.log(q_values / mixed))
    )


def compare_pattern_distributions(reference: Distribution, generated: Distribution, epsilon: float = 1e-8) -> dict[str, float]:
    """Compare reference and generated pattern distributions."""
    return {
        "pattern_kl_reference_to_generated": kl_divergence(reference, generated, epsilon),
        "pattern_kl_generated_to_reference": kl_divergence(generated, reference, epsilon),
        "pattern_js_reference_generated": js_divergence(reference, generated, epsilon),
    }


def _default_smb_slices_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "submodules"
        / "pcg_benchmark"
        / "pcg_benchmark"
        / "probs"
        / "smb"
        / "slices.txt"
    )


@lru_cache(maxsize=16)
def smb_reference_array(slices_path: str | None = None) -> np.ndarray:
    """Load the SMB benchmark reference slice vocabulary as a tile array."""
    path = Path(slices_path) if slices_path is not None else _default_smb_slices_path()
    slices = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not slices:
        return np.zeros((0, 0), dtype=np.int32)

    height = len(slices[0])
    data = np.zeros((height, len(slices)), dtype=np.int32)
    for x, slice_text in enumerate(slices):
        for y, char in enumerate(slice_text[:height]):
            data[y, x] = SMB_REFERENCE_CHAR_TILES.get(char, 0)
    return data


@lru_cache(maxsize=32)
def smb_reference_distribution(window: tuple[int, int] = (2, 2), slices_path: str | None = None) -> Distribution:
    """Return a tile-pattern distribution for the SMB reference corpus."""
    return normalize_counts(pattern_counts(smb_reference_array(slices_path), window))
