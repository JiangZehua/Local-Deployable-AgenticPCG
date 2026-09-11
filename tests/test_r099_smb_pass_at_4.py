from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.r099_smb_pass_at_4 import aggregate


def _sample(sample_index: int) -> dict:
    records = []
    for case_index in range(29):
        exact = case_index == sample_index
        gap = 0.0 if exact else 1.0
        records.append(
            {
                "case_id": f"case-{case_index:02d}",
                "targets": {
                    "complete": 1.0,
                    "enemies": 1,
                    "coins": 1,
                    "jumps": 1,
                },
                "exact": exact,
                "hard_valid": True,
                "e_inf": gap,
                "e_sum": gap * 3,
                "metric_gaps": {
                    "complete": 0.0,
                    "enemies": gap,
                    "coins": gap,
                    "jumps": gap,
                },
                "final_metrics": {
                    "complete": 1.0,
                    "enemies": int(exact),
                    "coins": int(exact),
                    "jumps": int(exact),
                },
                "final_level_sha256": f"{sample_index}-{case_index}",
                "costs": {
                    "llm_calls": 1,
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "latency_ms": 100,
                },
            }
        )
    return {
        "execution_complete": True,
        "contract_gate_pass": True,
        "aggregate": {"exact": 1},
        "records": records,
    }


class SmbPassAt4Tests(unittest.TestCase):
    def test_aggregate_uses_any_exact_sample_per_target(self) -> None:
        with TemporaryDirectory() as directory:
            paths = []
            for sample_index in range(4):
                path = Path(directory) / f"sample-{sample_index}.json"
                path.write_text(json.dumps(_sample(sample_index)))
                paths.append(path)
            payload = aggregate(paths)
        self.assertEqual(payload["aggregate"]["pass_at_4_targets"], 4)
        self.assertAlmostEqual(payload["aggregate"]["pass_at_4"], 4 / 29)
        self.assertEqual(payload["aggregate"]["single_trial_exact"], 4)
        self.assertEqual(payload["aggregate"]["costs"]["llm_calls"], 116)
        self.assertEqual(payload["aggregate"]["mean_unique_final_levels"], 4.0)

    def test_exactly_four_samples_are_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            aggregate([])


if __name__ == "__main__":
    unittest.main()
