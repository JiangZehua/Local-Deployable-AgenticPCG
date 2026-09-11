from ..generator import Generator
from . import TILES
import numpy as np

class MazeGenerator(Generator):
    class Cell:
        def __init__(self):
              self._walls = [True] * 4
              self.marked = False

        def unlockDirection(self, dx, dy):
            if dx == -1:
                self._walls[0] = False
            if dx == 1:
                self._walls[1] = False
            if dy == -1:
                self._walls[2] = False
            if dy == 1:
                self._walls[3] = False

        def getWall(self, dx, dy):
            if dx == -1:
                return self._walls[0]
            if dx == 1:
                return self._walls[1]
            if dy == -1:
                return self._walls[2]
            if dy == 1:
                return self._walls[3]
            return True
    
    def generate(self, level):
        height, width = np.array(level).shape
        maze_height, maze_width = height // 2, width // 2
        maze = []
        for y in range(maze_height):
            maze.append([])
            for x in range(maze_width):
                maze[y].append(MazeGenerator.Cell())
        start = {"x": np.random.randint(0, maze_width), "y": np.random.randint(0, maze_height)}
        open = [start]
        while len(open) > 0:
            np.random.shuffle(open)
            current = open.pop()
            if not maze[current["y"]][current["x"]].marked:
                surrounding = []
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (dx == 0 or dy == 0) and not (dx == 0 and dy == 0):
                            newPos = {"x": current["x"] + dx, "y": current["y"] + dy}
                            if (newPos["x"] >= 0 and newPos["y"] >= 0 and newPos["x"] <= maze_width - 1 and newPos["y"] <= maze_height - 1):
                                if maze[newPos["y"]][newPos["x"]].marked:
                                    surrounding.append({"x": dx, "y": dy})
                np.random.shuffle(surrounding)
                if len(surrounding) > 0:
                    maze[current["y"]][current["x"]].unlockDirection(surrounding[0]["x"], surrounding[0]["y"])
                    maze[current["y"]+surrounding[0]["y"]][current["x"]+surrounding[0]["x"]].unlockDirection(
                        -surrounding[0]["x"], -surrounding[0]["y"])
                maze[current["y"]][current["x"]].marked = True
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (dx == 0 or dy == 0) and not (dx == 0 and dy == 0):
                            newPos = {"x": current["x"] + dx, "y": current["y"] + dy}
                            if (newPos["x"] >= 0 and newPos["y"] >= 0 and newPos["x"] <= maze_width - 1 and newPos["y"] <= maze_height - 1):
                                open.append(newPos)
        result = np.array(level)
        for y in range(height):
            for x in range(width):
                if y % 2 == 0 and x % 2 == 0:
                    pos = {"x": x // 2, "y": y // 2}
                    result[y][x] = TILES["empty"]
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if abs(dx) + abs(dy) != 1:
                                continue
                            if not maze[pos["y"]][pos["x"]].getWall(dx, dy):
                                result[y + dy][x + dx] = TILES["empty"]
        if height % 2 == 0:
            for x in range(width):
                if result[height - 2][x] == TILES["solid"]:
                    result[height - 1][x] = TILES["solid"]
        if width % 2 == 0:
            for y in range(height):
                if result[y][width - 2] == TILES["solid"]:
                    result[y][width - 1] = TILES["solid"]
        return result
