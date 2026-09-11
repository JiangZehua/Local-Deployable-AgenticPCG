from ..generator import Generator
from . import TILES
import numpy as np

class DiggerGenerator(Generator):
    def __init__(self, change_prob=0.15, room_prob=0.01, room_size=3, stop_size=0.3):
        self._change_prob = change_prob
        self._room_prob = room_prob
        self._room_size = room_size
        self._stop_size = stop_size
        self._directions = [(-1,0), (1,0), (0,-1), (0,1)]

    def generate(self, level):
        height, width = np.array(level).shape
        level = np.full((height, width), TILES["solid"], dtype=int)
        x, y = np.random.randint(0, width), np.random.randint(0, height)
        dx, dy = self._directions[np.random.randint(len(self._directions))]
        while np.sum(level == TILES["empty"]) / (width * height) < self._stop_size:
            if np.random.rand() < self._change_prob:
                dx, dy = self._directions[np.random.randint(len(self._directions))]
            if np.random.rand() < self._room_prob:
                for rx in range(-self._room_size, self._room_size + 1):
                    for ry in range(-self._room_size, self._room_size + 1):
                        nx, ny = x + rx, y + ry
                        if 0 <= nx < width and 0 <= ny < height:
                            level[ny][nx] = TILES["empty"]
            else:
                level[y][x] = TILES["empty"]
            x += dx
            y += dy
            if x < 0 or x >= width or y < 0 or y >= height:
                x = np.clip(x, 0, width-1)
                y = np.clip(y, 0, height-1)
                dx, dy = self._directions[np.random.randint(len(self._directions))]
        return level
