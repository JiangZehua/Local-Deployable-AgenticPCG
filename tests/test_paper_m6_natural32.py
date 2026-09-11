import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/paper_m6_natural32.py"
MANIFEST = ROOT / "configs/paper_m6_natural32.json"
SPEC = importlib.util.spec_from_file_location("paper_m6_natural32", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class PaperM6Natural32Test(unittest.TestCase):
    def test_manifest_is_frozen_24_plus_8(self):
        payload = MODULE.load_manifest(MANIFEST)
        self.assertEqual(payload["status"], "frozen_before_generation")
        self.assertEqual(sum(c["role"] == "valid_target" for c in payload["cases"]), 24)
        self.assertEqual(sum(c["role"] == "guard" for c in payload["cases"]), 8)

    def test_ground_truth_is_manifest_owned(self):
        payload = json.loads(MANIFEST.read_text())
        for case in payload["cases"]:
            self.assertIn(case["expected"]["status"], {"resolved", "needs_clarification", "unsupported"})
            if case["role"] == "valid_target":
                self.assertEqual(set(case["expected"]["constraints"]), {"enemies", "coins", "jumps"})

    def test_rule_baseline_is_deterministic(self):
        payload = MODULE.load_manifest(MANIFEST)
        self.assertEqual(MODULE.baseline_evaluation(payload), MODULE.baseline_evaluation(payload))

    def test_rule_baseline_never_executes_guards(self):
        payload = MODULE.load_manifest(MANIFEST)
        result = MODULE.baseline_evaluation(payload)
        self.assertEqual(result["aggregate"]["false_resolve"], 0)


if __name__ == "__main__":
    unittest.main()
