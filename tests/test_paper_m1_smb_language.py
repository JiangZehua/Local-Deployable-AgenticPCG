from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.paper_m1_smb_language import (
    SEALED_GUARDS,
    SEALED_MANIFEST,
    VALIDATION_GUARDS,
    VALIDATION_MANIFEST,
    resolve_language_spec,
    structured_ceiling,
)


class FrozenManifestTests(unittest.TestCase):
    def test_static_playability_manifest_has_exact_structured_ceiling(self) -> None:
        manifest = Path(
            "benchmarks/smb/static_language.json"
        )
        frozen = json.loads(manifest.read_text())
        self.assertTrue(frozen["frozen"])
        self.assertEqual(frozen["valid_target_cases"], 7)
        self.assertEqual(frozen["guard_cases"], 1)
        with tempfile.TemporaryDirectory() as directory:
            result = structured_ceiling(
                manifest, Path(directory) / "static-ceiling.json"
            )
        self.assertEqual(result["cases"], 7)
        self.assertEqual(result["exact"], 7)
        self.assertTrue(result["gate_pass"])

    def test_validation_and_sealed_manifests_are_frozen_before_calls(self) -> None:
        validation = json.loads(VALIDATION_MANIFEST.read_text())
        sealed = json.loads(SEALED_MANIFEST.read_text())
        self.assertTrue(validation["frozen"])
        self.assertTrue(sealed["frozen"])
        self.assertEqual(validation["valid_target_cases"], 29)
        self.assertEqual(validation["guard_cases"], len(VALIDATION_GUARDS))
        self.assertEqual(sealed["valid_target_cases"], 38)
        self.assertEqual(sealed["guard_cases"], len(SEALED_GUARDS))
        self.assertFalse(validation["witness_levels_or_traces_loaded"])
        self.assertFalse(sealed["witness_levels_or_traces_loaded"])
        self.assertEqual(
            len({row["instruction"] for row in validation["cases"]}),
            len(validation["cases"]),
        )

    def test_structured_validation_ceiling_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = structured_ceiling(
                VALIDATION_MANIFEST, Path(directory) / "ceiling.json"
            )
        self.assertEqual(result["cases"], 29)
        self.assertEqual(result["exact"], 29)
        self.assertTrue(result["gate_pass"])

    def test_exposed_manifest_retains_eighteen_valid_and_six_guard_roles(self) -> None:
        exposed = json.loads(
            Path(
                "benchmarks/smb/language.json"
            ).read_text()
        )
        valid = [
            row for row in exposed["cases"]
            if row["expected"]["status"] == "resolved"
        ]
        guards = [
            row for row in exposed["cases"]
            if row["expected"]["status"] != "resolved"
        ]
        self.assertEqual(len(valid), 18)
        self.assertEqual(len(guards), 6)

    def test_exposed_search_controls_reach_paper_resolver(self) -> None:
        exposed = json.loads(
            Path(
                "benchmarks/smb/language.json"
            ).read_text()
        )
        case = next(
            row for row in exposed["cases"]
            if row["case_id"] == "dynamic-search-jump-08"
        )
        result = resolve_language_spec(case["expected"], case)
        self.assertEqual(result.status, "exact")
        self.assertEqual(result.source, "bounded_search")
        self.assertEqual(result.strategy, "jump_episode_search")


if __name__ == "__main__":
    unittest.main()
