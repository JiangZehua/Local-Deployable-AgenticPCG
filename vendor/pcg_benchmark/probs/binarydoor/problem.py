from pcg_benchmark.probs import Problem
from pcg_benchmark.spaces import ArraySpace, IntegerSpace, DictionarySpace
from pcg_benchmark.probs.utils import get_range_reward, get_number_regions
import numpy as np
from PIL import Image
from enum import IntEnum
import os
from collections import deque


class Tile(IntEnum):
    EMPTY = 0
    WALL = 1


class BinaryDoorProblem(Problem):
    """Binary maze with two door openings in the wall border.

    The objective is to connect the two doors through the maze AND hit a
    target path length metric (``door_path``).  Doors are randomly placed on
    the non-corner border of the augmented (H+2)x(W+2) grid with a minimum
    perimeter distance to avoid trivial configurations.
    """

    def __init__(self, **kwargs):
        Problem.__init__(self, **kwargs)
        self._width = kwargs.get("width")
        self._height = kwargs.get("height")
        self._max_path = int(np.ceil(self._width * self._height / 2) + max(self._width, self._height))
        self._target = kwargs.get("door_path", self._max_path // 2)
        self._door_seed = kwargs.get("door_seed", 42)
        self._diversity = kwargs.get("diversity", 0.4)

        self._cerror = max(int(self._target * 0.1), 1)
        self._content_space = ArraySpace((self._height, self._width), IntegerSpace(2))

        min_path = min(self._target + self._cerror, self._max_path - 1)
        self._control_space = DictionarySpace({"door_path": IntegerSpace(min_path, self._max_path)})

        # Generate deterministic door positions
        self._door1, self._door2 = self._generate_doors()

    def _generate_doors(self):
        """Generate two door positions on the non-corner border of the augmented grid.

        Returns two (row, col) positions in the augmented (H+2)x(W+2) grid.
        Minimum perimeter distance >= min(W, H) to prevent trivial configs.
        """
        ah = self._height + 2  # augmented height
        aw = self._width + 2   # augmented width

        # Enumerate non-corner border positions as an ordered perimeter cycle
        # Top row (left to right, excluding corners)
        border = []
        for c in range(1, aw - 1):
            border.append((0, c))
        # Right column (top to bottom, excluding corners)
        for r in range(1, ah - 1):
            border.append((r, aw - 1))
        # Bottom row (right to left, excluding corners)
        for c in range(aw - 2, 0, -1):
            border.append((ah - 1, c))
        # Left column (bottom to top, excluding corners)
        for r in range(ah - 2, 0, -1):
            border.append((r, 0))

        perimeter_len = len(border)
        min_dist = min(self._width, self._height)

        rng = np.random.RandomState(self._door_seed)

        # Pick door1 randomly
        idx1 = rng.randint(0, perimeter_len)
        door1 = border[idx1]

        # Find candidates for door2 with perimeter distance >= min_dist
        candidates = []
        for i, pos in enumerate(border):
            # Perimeter distance (shortest walk around the perimeter)
            d_forward = (i - idx1) % perimeter_len
            d_backward = (idx1 - i) % perimeter_len
            peri_dist = min(d_forward, d_backward)
            if peri_dist >= min_dist:
                candidates.append((i, peri_dist))

        if candidates:
            # Pick randomly among valid candidates
            ci = rng.randint(0, len(candidates))
            idx2 = candidates[ci][0]
        else:
            # Fallback: pick the farthest position
            best_dist = -1
            best_idx = (idx1 + perimeter_len // 2) % perimeter_len
            for i in range(perimeter_len):
                if i == idx1:
                    continue
                d_forward = (i - idx1) % perimeter_len
                d_backward = (idx1 - i) % perimeter_len
                peri_dist = min(d_forward, d_backward)
                if peri_dist > best_dist:
                    best_dist = peri_dist
                    best_idx = i
            idx2 = best_idx

        door2 = border[idx2]
        return door1, door2

    def _build_augmented(self, content):
        """Build augmented grid: pad content with walls, then open doors."""
        content = np.array(content)
        augmented = np.pad(content, 1, constant_values=int(Tile.WALL))
        augmented[self._door1[0], self._door1[1]] = int(Tile.EMPTY)
        augmented[self._door2[0], self._door2[1]] = int(Tile.EMPTY)
        return augmented

    def _bfs_path(self, augmented, start, end):
        """BFS from start to end on the augmented grid.

        Returns (path_length, distance_map).  path_length is 0 if no path.
        Distance map has -1 for unvisited cells.
        """
        h, w = augmented.shape
        d_map = np.full((h, w), -1, dtype=np.int32)

        if augmented[start[0], start[1]] != int(Tile.EMPTY):
            return 0, d_map
        if augmented[end[0], end[1]] != int(Tile.EMPTY):
            return 0, d_map

        d_map[start[0], start[1]] = 0
        queue = deque([start])

        while queue:
            r, c = queue.popleft()
            if (r, c) == end:
                return int(d_map[r, c]), d_map

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    if d_map[nr, nc] == -1 and augmented[nr, nc] == int(Tile.EMPTY):
                        d_map[nr, nc] = d_map[r, c] + 1
                        queue.append((nr, nc))

        return 0, d_map

    def info(self, content):
        content = np.array(content)
        augmented = self._build_augmented(content)
        number_regions = get_number_regions(content, [Tile.EMPTY])
        door_path, d_map = self._bfs_path(augmented, self._door1, self._door2)

        return {
            "door_path": door_path,
            "regions": number_regions,
            "flat": content.flatten(),
            "d_map": d_map,
            "door1": self._door1,
            "door2": self._door2,
        }

    def quality(self, info):
        number_regions = get_range_reward(info["regions"], 0, 1, 1, self._width * self._height / 10)
        door_path_reward = get_range_reward(info["door_path"], 0, self._target, self._max_path)
        return (number_regions + door_path_reward) / 2

    def diversity(self, info1, info2):
        hamming = abs(info1["flat"] - info2["flat"]).sum()
        return get_range_reward(hamming, 0, self._diversity * self._width * self._height, self._width * self._height)

    def controlability(self, info, control):
        cerror = max(int(control["door_path"] * 0.1), 1)
        door_path = get_range_reward(
            info["door_path"], 0,
            control["door_path"] - cerror,
            control["door_path"] + cerror,
            self._max_path,
        )
        return door_path

    def _backtrack_path(self, d_map, end):
        """Follow decreasing distances from end back to 0."""
        path = [end]
        current = end
        current_val = d_map[end[0], end[1]]
        if current_val <= 0:
            return path

        while current_val > 0:
            r, c = current
            found = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < d_map.shape[0] and 0 <= nc < d_map.shape[1]:
                    if d_map[nr, nc] == current_val - 1:
                        current = (nr, nc)
                        path.append(current)
                        current_val -= 1
                        found = True
                        break
            if not found:
                break

        return path

    def render(self, content, info=None):
        scale = 16
        graphics = [
            Image.open(os.path.join(os.path.dirname(__file__), "images/empty.png")).convert('RGBA'),  # 0 empty
            Image.open(os.path.join(os.path.dirname(__file__), "images/solid.png")).convert('RGBA'),  # 1 wall
            Image.open(os.path.join(os.path.dirname(__file__), "images/path.png")).convert('RGBA'),   # 2 path
            Image.open(os.path.join(os.path.dirname(__file__), "images/solid.png")).convert('RGBA'),  # 3 border
            Image.open(os.path.join(os.path.dirname(__file__), "images/door.png")).convert('RGBA'),   # 4 door
        ]

        content = np.array(content)
        # Build augmented with border=3, doors=4
        padded = np.pad(content, 1, constant_values=3)
        padded[self._door1[0], self._door1[1]] = 4
        padded[self._door2[0], self._door2[1]] = 4

        lvl_image = Image.new("RGBA", (padded.shape[1] * scale, padded.shape[0] * scale), (0, 0, 0, 255))

        # Draw base tiles
        for y in range(padded.shape[0]):
            for x in range(padded.shape[1]):
                tile_idx = padded[y, x]
                lvl_image.paste(graphics[tile_idx], (x * scale, y * scale, (x + 1) * scale, (y + 1) * scale))

        # Draw path from door1 to door2 if info available
        if info and "d_map" in info and info.get("door_path", 0) > 0:
            d_map = info["d_map"]
            path_coords = self._backtrack_path(d_map, self._door2)
            for r, c in path_coords:
                # d_map is in augmented coordinates, which match padded coordinates
                lvl_image.paste(graphics[2], (c * scale, r * scale, (c + 1) * scale, (r + 1) * scale), mask=graphics[2])

        return lvl_image
