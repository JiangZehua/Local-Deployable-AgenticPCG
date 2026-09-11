from ..generator import Generator
from .wfc import NumpyOverlappingModel
from . import TILES
import numpy as np
import os
from PIL import Image

class WFCGenerator(Generator):
    def __init__(self, img_name="maze", pattern_size=3, trials=10):
        with Image.open(os.path.join(os.path.dirname(__file__), f"imgs/{img_name}.png")) as img:
            self._img = np.array(img)
        self._pattern_size = pattern_size
        self._trials = trials

    def generate(self, level):
        height, width = np.array(level).shape
        model = NumpyOverlappingModel(width, height, self._img, N_value=self._pattern_size)
        finished = False
        trials = 0
        while not finished and trials < self._trials:
            finished = model.Run(np.random.random(), 0)
            trials += 1
        if finished:
            new_level = np.array(model.Graphics())
            for y in range(height):
                for x in range(width):
                    if new_level[y][x][0] == 0:
                        level[y][x] = TILES["solid"]
                    else:
                        level[y][x] = TILES["empty"]
        return np.array(level)
