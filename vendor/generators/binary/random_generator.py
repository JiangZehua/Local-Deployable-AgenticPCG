from ..generator import Generator
from . import TILES
import numpy as np

class RandomGenerator(Generator):
    def __init__(self, solid_prob=0.5):
        self._tile_probs = {
            "solid": solid_prob,
            "empty": 1 - solid_prob
        }

    def generate(self, level):
        height, width = np.array(level).shape
        level = np.random.choice([TILES["solid"], TILES["empty"]], size=(height, width), p=[self._tile_probs["solid"], self._tile_probs["empty"]])
        return level
