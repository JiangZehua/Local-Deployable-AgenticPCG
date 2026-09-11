from ..generator import Generator
from . import TILES
import numpy as np

class ConnectGenerator(Generator):
    def __init__(self, smallest_region_size=5):
        self._smallest_region_size = smallest_region_size

    def _get_empty(self, level):
        tiles=[]
        for y in range(level.shape[0]):
            for x in range(level.shape[1]):
                if level[y][x] == TILES["empty"]:
                    tiles.append((x,y))
        return tiles
    
    def _get_regions(self, level):
        empty_tiles = self._get_empty(level)
        np.random.shuffle(empty_tiles)
        color_map = np.full(level.shape, -1)
        
        color_index = -1
        while len(empty_tiles) > 0:
            queue = [empty_tiles.pop()]
            color_index += 1
            while len(queue) > 0:
                (cx, cy) = queue.pop(0)
                if color_map[cy][cx] != -1 or level[cy][cx] == TILES["solid"]:
                    continue
                color_map[cy][cx] = color_index
                for (dx,dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx,ny=cx+dx,cy+dy
                    if nx < 0 or ny < 0 or nx >= len(level[0]) or ny >= len(level):
                        continue
                    queue.append((nx, ny))
        regions = []
        for i in range(color_index + 1):
            region_tiles = []
            for y in range(level.shape[0]):
                for x in range(level.shape[1]):
                    if color_map[y][x] == i:
                        region_tiles.append((x,y))
            if len(region_tiles) >= self._smallest_region_size:
                regions.append(region_tiles)
            else:
                for (x,y) in region_tiles:
                    level[y][x] = TILES["solid"]
        return regions
    
    def _get_corridor(self, x1, y1, x2, y2):
        points = []
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
        level = np.array(level)
        regions = self._get_regions(level)
        for i in range(len(regions) - 1):
            region1 = regions[i]
            region2 = regions[i + 1]
            x1, y1 = region1[np.random.randint(len(region1))]
            x2, y2 = region2[np.random.randint(len(region2))]
            corridors = self._get_corridor(x1, y1, x2, y2)
            for (x, y) in corridors:
                level[y][x] = TILES["empty"]
        return level
