from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from experiments.zelda_bsp_open_prompt_eval import room_distribution
from toolusepcg.level import Level
from toolusepcg.tools.generate_zelda_wall import (
    GenerateZeldaBSPTool,
    GenerateZeldaDiggerTool,
)


class _FakeEditManager:
    def __init__(self, data: np.ndarray):
        self.data = data.copy()

    @property
    def level_data(self) -> np.ndarray:
        return self.data.copy()

    @property
    def level(self) -> Level:
        return Level(self.data.copy(), problem_type="zelda")

    def set_tiles(self, coords, value):
        changed = 0
        diff = []
        for row, col in coords:
            old = int(self.data[row, col])
            if old != value:
                self.data[row, col] = value
                changed += 1
                diff.append({"y": row, "x": col, "old": old, "new": value})
        return SimpleNamespace(num_tiles_changed=changed, diff=diff, errors=[])


class ZeldaBspOpenPromptEvalTests(unittest.TestCase):
    def test_bsp_adapter_preserves_every_special_tile(self) -> None:
        data = np.ones((32, 32), dtype=int)
        data[0, :] = data[-1, :] = 0
        data[:, 0] = data[:, -1] = 0
        specials = {(2, 2): 2, (10, 10): 3, (28, 28): 4, (5, 18): 5}
        for cell, value in specials.items():
            data[cell] = value
        manager = _FakeEditManager(data)
        np.random.seed(42)

        result = GenerateZeldaBSPTool(manager).execute(splits=4, min_width=5, min_height=5)

        self.assertTrue(result.success)
        self.assertGreater(result.data["num_tiles_changed"], 0)
        for cell, value in specials.items():
            self.assertEqual(int(manager.data[cell]), value)
            row, col = cell
            self.assertTrue(
                any(
                    manager.data[nrow, ncol] == 1
                    for nrow, ncol in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                )
            )
        self.assertTrue(set(np.unique(manager.data)) <= {0, 1, 2, 3, 4, 5})

    def test_digger_transaction_connects_every_preserved_entity(self) -> None:
        data = np.ones((32, 32), dtype=int)
        data[0, :] = data[-1, :] = 0
        data[:, 0] = data[:, -1] = 0
        specials = {(2, 2): 2, (10, 10): 3, (28, 28): 4, (5, 18): 5}
        for cell, value in specials.items():
            data[cell] = value
        manager = _FakeEditManager(data)
        np.random.seed(42)

        result = GenerateZeldaDiggerTool(manager).execute(stop_size=0.3)

        self.assertTrue(result.success)
        for cell, value in specials.items():
            self.assertEqual(int(manager.data[cell]), value)
            row, col = cell
            self.assertTrue(
                any(
                    manager.data[nrow, ncol] == 1
                    for nrow, ncol in (
                        (row - 1, col),
                        (row + 1, col),
                        (row, col - 1),
                        (row, col + 1),
                    )
                )
            )

    def test_room_readout_counts_one_enemy_per_room_and_key(self) -> None:
        level = [list("#" * 15)]
        for _ in range(13):
            level.append(list("#" + "." * 13 + "#"))
        level.append(list("#" * 15))
        for row in range(1, 14):
            level[row][7] = "#"
        for col in range(1, 14):
            level[7][col] = "#"
        for row, col in ((7, 3), (7, 11), (3, 7)):
            level[row][col] = "."
        for row, col in ((3, 3), (3, 11), (11, 3), (11, 11)):
            level[row][col] = "E"
        level[4][4] = "K"

        result = room_distribution(["".join(row) for row in level])

        self.assertEqual(result["room_count"], 4)
        self.assertEqual(result["enemy_counts_by_room"], [1, 1, 1, 1])
        self.assertTrue(result["all_rooms_have_exactly_one_enemy"])
        self.assertTrue(result["key_is_in_a_room"])


if __name__ == "__main__":
    unittest.main()
