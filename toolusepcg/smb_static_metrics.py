"""Tile-count and support metrics for the ``smb_static`` domain.

These functions remain pure grid inspection.  The static evaluator combines
them with an A* gameplay witness before declaring a level valid.  A pipe is one
filled, exactly two-column-wide connected rectangle with terrain directly
beneath both columns; enemies likewise require solid support below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .level import SMBTiles


STATIC_TARGET_METRICS = (
    "pipe_count",
    "enemy_tile_count",
    "coin_tile_count",
    "block_tile_count",
)


@dataclass(frozen=True)
class PipeSummary:
    count: int
    malformed_components: int
    unsupported_components: int


SUPPORT_TILES = frozenset(
    {
        int(SMBTiles.SOLID),
        int(SMBTiles.PLATFORM),
        int(SMBTiles.BRICK),
        int(SMBTiles.QUESTION),
        int(SMBTiles.TUBE),
    }
)


def build_static_support_scaffold(height: int = 16, width: int = 32) -> np.ndarray:
    """Return an empty level with a fixed, non-target platform floor."""
    if height < 2 or width < 1:
        raise ValueError("SMB support scaffold requires height >= 2 and width >= 1")
    grid = np.full((height, width), int(SMBTiles.EMPTY), dtype=np.int32)
    grid[-1, :] = int(SMBTiles.PLATFORM)
    return grid


def _pipe_components(grid: np.ndarray) -> PipeSummary:
    tube = int(SMBTiles.TUBE)
    height, width = grid.shape
    seen: set[tuple[int, int]] = set()
    valid_count = 0
    malformed = 0
    unsupported = 0

    for start_y in range(height):
        for start_x in range(width):
            if int(grid[start_y, start_x]) != tube or (start_y, start_x) in seen:
                continue
            stack = [(start_y, start_x)]
            seen.add((start_y, start_x))
            component: set[tuple[int, int]] = set()
            while stack:
                y, x = stack.pop()
                component.add((y, x))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and int(grid[ny, nx]) == tube
                        and (ny, nx) not in seen
                    ):
                        seen.add((ny, nx))
                        stack.append((ny, nx))

            ys = [coord[0] for coord in component]
            xs = [coord[1] for coord in component]
            min_y, max_y = min(ys), max(ys)
            min_x, max_x = min(xs), max(xs)
            expected = {
                (y, x)
                for y in range(min_y, max_y + 1)
                for x in range(min_x, max_x + 1)
            }
            if max_x - min_x + 1 == 2 and component == expected:
                valid_count += 1
                support_y = max_y + 1
                if support_y >= height or any(
                    int(grid[support_y, x]) not in SUPPORT_TILES
                    for x in range(min_x, max_x + 1)
                ):
                    unsupported += 1
            else:
                malformed += 1

    return PipeSummary(valid_count, malformed, unsupported)


def evaluate_static_grid(data: np.ndarray) -> dict[str, int | float]:
    """Return deterministic tile counts and support checks."""
    grid = np.asarray(data)
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D SMB grid, got shape {grid.shape}")

    pipes = _pipe_components(grid)
    enemy_mask = np.isin(
        grid,
        [int(SMBTiles.GOOMBA), int(SMBTiles.KOOPA), int(SMBTiles.SPINY)],
    )
    enemy_positions = np.argwhere(enemy_mask)
    unsupported_enemies = sum(
        int(y) + 1 >= grid.shape[0]
        or int(grid[int(y) + 1, int(x)]) not in SUPPORT_TILES
        for y, x in enemy_positions
    )
    enemy_count = int(enemy_mask.sum())
    return {
        "pipe_count": pipes.count,
        "enemy_tile_count": enemy_count,
        "coin_tile_count": int(np.sum(grid == int(SMBTiles.COIN))),
        "block_tile_count": int(
            np.isin(
                grid,
                [int(SMBTiles.SOLID), int(SMBTiles.BRICK), int(SMBTiles.QUESTION)],
            ).sum()
        ),
        "tube_issues": pipes.malformed_components,
        "unsupported_pipe_count": pipes.unsupported_components,
        "unsupported_enemy_count": int(unsupported_enemies),
        "enemy_support_ratio": (
            1.0 if enemy_count == 0 else (enemy_count - unsupported_enemies) / enemy_count
        ),
        "empty_ratio": float(np.mean(grid == int(SMBTiles.EMPTY))),
    }
