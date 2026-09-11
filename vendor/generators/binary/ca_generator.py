from ..generator import Generator
from . import TILES
import numpy as np

class CAGenerator(Generator):
    def __init__(self, iterations=10, solid_count=2, empty_count=6):
        self._iterations = iterations
        self._solid_count = solid_count
        self._empty_count = empty_count
        

    def generate(self, level):
        height, width = np.array(level).shape
        for _ in range(self._iterations):
            new_level = level.copy()
            for y in range(height):
                for x in range(width):
                    solid_neighbors = 0
                    empty_neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if level[ny][nx] == TILES["solid"]:
                                    solid_neighbors += 1
                                else:
                                    empty_neighbors += 1
                            else:
                                solid_neighbors += 1  # Treat out-of-bounds as solid
                    if solid_neighbors <= self._solid_count:
                        new_level[y][x] = TILES["solid"]
                    if empty_neighbors >= self._empty_count:
                        new_level[y][x] = TILES["empty"]
            level = new_level
        return np.array(level)