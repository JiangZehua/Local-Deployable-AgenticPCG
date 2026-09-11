import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments/paper_m7_fixed_rng36.py"
MANIFEST = ROOT / "configs/paper_m7_fixed_rng36.yaml"
SPEC = importlib.util.spec_from_file_location("paper_m7_fixed_rng36", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class PaperM7FixedRng36Test(unittest.TestCase):
    def test_manifest_pairs_rng_within_family(self):
        payload = MODULE.load_manifest(MANIFEST)
        for family in {task["family_id"] for task in payload["tasks"]}:
            rows = [task for task in payload["tasks"] if task["family_id"] == family]
            self.assertEqual(len({task["seed"] for task in rows}), 1)
            self.assertEqual(len({task.get("initial", payload["default_initial"]) for task in rows}), 1)

    def test_episode_denominator_is_36(self):
        payload = MODULE.load_manifest(MANIFEST)
        self.assertEqual(len(payload["tasks"]) * len(payload["arms"]), 36)

    def test_keyword_baseline_is_frozen_and_complete(self):
        payload = MODULE.load_manifest(MANIFEST)
        baseline = MODULE.keyword_baseline(payload)
        self.assertEqual(baseline["aggregate"]["cases"], 18)
        self.assertEqual(baseline["aggregate"]["route_exact"], 18)

    def test_process_seed_wrapper_repeats_python_and_numpy(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("NumPy is supplied by the frozen inference environment")
        wrapper_path = ROOT / "experiments/run_seeded_entrypoint.py"
        wrapper_spec = importlib.util.spec_from_file_location("run_seeded_entrypoint", wrapper_path)
        wrapper = importlib.util.module_from_spec(wrapper_spec)
        assert wrapper_spec and wrapper_spec.loader
        wrapper_spec.loader.exec_module(wrapper)
        self.assertEqual(wrapper.seed_process(6501), wrapper.seed_process(6501))

    def test_manual_equivalent_requires_trace_level_preservation(self):
        task = {"family_id": "reconnect_regions"}
        record = {
            "routing": {"accepted_macro_route": False},
            "structure": {"regions": 1, "functional_valid": True},
        }
        initial = "#####\n#P#K#\n#E#D#\n#####"
        final = "#####\n#P.K#\n#E#D#\n#####"
        events = [
            {"event": "initial_level", "level": initial, "metrics": {"regions": 2}},
            {
                "event": "step",
                "step_num": 1,
                "accepted": True,
                "parsed_message": {"tool_calls": [{"tool_name": "place_tile"}]},
                "tool_results": [{"result": {"success": True, "data": {"diff": [{"old": "wall", "new": "empty"}]}}}],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
            (root / "final_level.txt").write_text(final + "\n")
            self.assertTrue(MODULE._strict_reconnect_equivalent(task, root, record)["verified"])
            events[1]["tool_results"][0]["result"]["data"]["diff"] = [{"old": "empty", "new": "wall"}]
            (root / "trace.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")
            self.assertFalse(MODULE._strict_reconnect_equivalent(task, root, record)["verified"])


if __name__ == "__main__":
    unittest.main()
