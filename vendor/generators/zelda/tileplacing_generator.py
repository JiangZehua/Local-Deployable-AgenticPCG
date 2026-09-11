from ..generator import Generator
from . import TILES
import numpy as np

class TilePlacingGenerator(Generator):
    def generate(self, level):
        height, width = level.shape
        level = np.full((height, width), TILES["empty"], dtype=int)
        
        all_positions = [(x, y) for x in range(width) for y in range(height)]
        np.random.shuffle(all_positions)
        empty_positions = []

        # Placing Enemies and Solid Tiles Randomly
        solid_prob = 0.2
        solid_max_length = min(width, height) // 4
        enemy_prob = 0.05
        for x, y in all_positions:
            if level[y][x] != TILES["empty"]:
                continue
            if np.random.rand() < enemy_prob:
                level[y][x] = TILES["enemy"]
            elif np.random.rand() < solid_prob:
                length = np.random.randint(1, solid_max_length + 1)
                direction = [(-1,0), (1,0), (0,-1), (0,1)][np.random.randint(0,4)]
                for i in range(length):
                    nx, ny = x + i * direction[0], y + i * direction[1]
                    if  nx < 0 or nx >= width or ny < 0 or ny >= height or level[ny][nx] != TILES["empty"]:
                        break
                    level[ny, nx] = TILES["solid"]
            else:
                empty_positions.append((x, y))

        # Placing Player, Key, Door in the empty spaces
        np.random.shuffle(empty_positions)
        player_pos = empty_positions.pop()
        key_pos = empty_positions.pop()
        door_pos = empty_positions.pop()
        level[player_pos[1]][player_pos[0]] = TILES["player"]
        level[key_pos[1]][key_pos[0]] = TILES["key"]
        level[door_pos[1]][door_pos[0]] = TILES["door"]

        return level