from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments import high_level_freeform_eval as freeform


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "configs/high_level_freeform_smb_eval.yaml"
ZELDA_SUITE_PATH = ROOT / "configs/high_level_freeform_zelda_27b_pilot.yaml"


def blank_level() -> list[list[str]]:
    return [list("." * 32) for _ in range(16)]


def rows(level: list[list[str]]) -> list[str]:
    return ["".join(row) for row in level]


def blank_zelda() -> list[list[str]]:
    level = [list("." * 16) for _ in range(16)]
    for index in range(16):
        level[0][index] = level[15][index] = "#"
        level[index][0] = level[index][15] = "#"
    return level


class HighLevelFreeformEvalTests(unittest.TestCase):




    def test_t1_exact_hollow_frame(self) -> None:
        level = blank_level()
        for row in range(3, 9):
            for col in range(4, 16):
                if row in {3, 8} or col in {4, 15}:
                    level[row][col] = "B"
        metrics = freeform.t1_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))
        self.assertEqual(metrics["best_frame"]["frame_iou"], 1.0)

    def test_t2_exact_smb_templates(self) -> None:
        level = blank_level()
        left = 2
        for letter in ("S", "M", "B"):
            template = freeform.LETTER_TEMPLATES[letter]
            width = max(col for _, col in template) + 1
            for row, col in template:
                level[row + 3][col + left] = "B"
            left += width + 2
        metrics = freeform.t2_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))
        self.assertEqual(metrics["glyph_regions"], 3)

    def test_t3_staircase_entity_relations(self) -> None:
        level = blank_level()
        platforms = ((12, 2, 6), (9, 10, 14), (6, 18, 22))
        for row, left, right in platforms:
            for col in range(left, right + 1):
                level[row][col] = "B"
        level[11][4] = "G"
        level[5][20] = "K"
        metrics = freeform.t3_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))
        self.assertEqual(metrics["matching_staircase_count"], 1)

    def test_t4_tower_and_koopa_top(self) -> None:
        level = blank_level()
        for row in range(3, 14):
            level[row][10] = "B"
            level[row][15] = "B"
        for col in range(10, 16):
            level[3][col] = "B"
        level[2][12] = "K"
        metrics = freeform.t4_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))
        self.assertEqual(metrics["tower_height"], 11)

    def test_zelda_t1_u_courtyard(self) -> None:
        level = blank_zelda()
        for row in range(4, 12):
            level[row][4] = level[row][11] = "#"
        for col in range(4, 12):
            level[11][col] = "#"
        metrics = freeform.zelda_t1_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))

    def test_zelda_t2_uppercase_z(self) -> None:
        level = blank_zelda()
        for row, col in freeform.Z_TEMPLATE:
            level[row + 4][col + 5] = "#"
        metrics = freeform.zelda_t2_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))
        self.assertEqual(metrics["z_template_iou"], 1.0)

    def test_zelda_t3_composition(self) -> None:
        level = blank_zelda()
        level[3][2] = "P"
        level[5][7] = "K"
        level[4][12] = "D"
        for col in (6, 7, 8):
            level[9][col] = "E"
        metrics = freeform.zelda_t3_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))

    def test_zelda_t4_guarded_vault(self) -> None:
        level = blank_zelda()
        for row in range(6, 13):
            level[row][4] = level[row][11] = "#"
        for col in range(4, 12):
            level[12][col] = "#"
        level[9][7] = "K"
        level[6][7] = "D"
        for col in (6, 7, 8):
            level[4][col] = "E"
        metrics = freeform.zelda_t4_metrics(rows(level))
        self.assertTrue(all(metrics["atoms"].values()))

    def test_zelda_gated_courtyard_accepts_one_deliberate_entrance(self) -> None:
        level = [list("." * 24) for _ in range(24)]
        for index in range(24):
            level[0][index] = level[23][index] = "#"
            level[index][0] = level[index][23] = "#"
        for row in range(7, 19):
            level[row][5] = level[row][18] = "#"
        for col in range(5, 19):
            level[7][col] = level[18][col] = "#"
        level[7][11] = "D"
        level[7][12] = "."
        level[12][12] = "K"
        level[2][2] = "P"
        for col in (10, 11, 12):
            level[5][col] = "E"

        metrics = freeform.zelda_gated_courtyard_metrics(rows(level))

        self.assertTrue(all(metrics["atoms"].values()))
        self.assertEqual(metrics["best_gated_courtyard"]["gate_cells"], [(7, 11), (7, 12)])

    def test_zelda_gated_courtyard_rejects_two_separate_gaps(self) -> None:
        level = [list("." * 24) for _ in range(24)]
        for index in range(24):
            level[0][index] = level[23][index] = "#"
            level[index][0] = level[index][23] = "#"
        for row in range(7, 19):
            level[row][5] = level[row][18] = "#"
        for col in range(5, 19):
            level[7][col] = level[18][col] = "#"
        level[7][9] = "."
        level[7][14] = "D"
        level[12][12] = "K"
        level[2][2] = "P"
        for col in (10, 11, 12):
            level[5][col] = "E"

        metrics = freeform.zelda_gated_courtyard_metrics(rows(level))

        self.assertFalse(metrics["atoms"]["one_contiguous_passable_gate"])

    def test_zelda_functional_contract(self) -> None:
        metrics = {
            "players": 1,
            "keys": 1,
            "doors": 1,
            "enemies": 3,
            "player_key": 12,
            "key_door": 10,
            "regions": 1,
        }
        self.assertTrue(freeform._functional_result("zelda", metrics, True)["success"])
        metrics["regions"] = 2
        self.assertFalse(freeform._functional_result("zelda", metrics, True)["success"])


    def test_exact_sign_flip_resolution_at_three_pairs(self) -> None:
        self.assertEqual(freeform.exact_sign_flip_p([1.0, 1.0, 1.0]), 0.25)
        self.assertEqual(freeform.exact_sign_flip_p([0.0, 0.0, 0.0]), 1.0)


if __name__ == "__main__":
    unittest.main()
