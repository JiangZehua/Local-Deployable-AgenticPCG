from ..generator import Generator
from . import TILES
import numpy as np

class BSPGenerator(Generator):
    def __init__(self, splits=3, min_wdith=5, min_height=5):
        self._splits = splits
        self._min_width = min_wdith
        self._min_height = min_height
    
    def _split_space(self, x, y, width, height):
        rectangles = []

        if width / 2 >= self._min_width or height / 2 >= self._min_height:
            if width >= 2 * self._min_width and (height < 2 * self._min_height or np.random.choice([True, False])):

                split_x = np.random.randint(self._min_width, width - self._min_width)
                rect1 = {"x": x, "y": y, "width": split_x, "height": height}
                rect2 = {"x": x + split_x, "y": y, "width": width - split_x, "height": height}
            else:
                # Split horizontally
                split_y = np.random.randint(self._min_height, height - self._min_height)
                rect1 = {"x": x, "y": y, "width": width, "height": split_y}
                rect2 = {"x": x, "y": y + split_y, "width": width, "height": height - split_y}

            rectangles.extend([rect1, rect2])
        else:
            rectangles.append({"x": x, "y": y, "width": width, "height": height})

        return rectangles
    
    def _get_corridor(self, room1, room2):
        points = []
        # Get centers
        x1, y1 = room1["x"] + room1["width"] // 2, room1["y"] + room1["height"] // 2
        x2, y2 = room2["x"] + room2["width"] // 2, room2["y"] + room2["height"] // 2
        if np.random.choice([True, False]):
            # Horizontal first, then vertical
            for x in range(min(x1, x2), max(x1, x2) + 1):
                points.append((x, y1))
            for y in range(min(y1, y2), max(y1, y2) + 1):
                points.append((x2, y))
        else:
            # Vertical first, then horizontal
            for y in range(min(y1, y2), max(y1, y2) + 1):
                points.append((x1, y))
            for x in range(min(x1, x2), max(x1, x2) + 1):
                points.append((x, y2))

        return points

    def generate(self, level):
        height, width = np.array(level).shape
        level = np.full((height, width), TILES["solid"], dtype=int)
        outer_rect = {"x": 0, "y": 0, "width": width, "height": height}
        rects = [outer_rect]
        for _ in range(self._splits):
            new_rects = []
            for rect in rects:
                new_rects.extend(self._split_space(rect["x"], rect["y"], rect["width"], rect["height"]))
            rects = new_rects
        for rect in rects:
            for y in range(rect["y"] + 1, rect["y"] + rect["height"] - 1):
                for x in range(rect["x"] + 1, rect["x"] + rect["width"] - 1):
                    level[y][x] = TILES["empty"]
        for i in range(len(rects) - 1):
            corridors = self._get_corridor(rects[i], rects[i + 1])
            for (x, y) in corridors:
                level[y][x] = TILES["empty"]
        return level