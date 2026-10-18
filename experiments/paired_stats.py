"""Paired perfect-match statistics with the frozen Z64 bootstrap settings.

The default metric scores the final saved map. ``error_free`` uses the
record's ``exact`` field, which also requires no unrecovered parse or tool
errors during the episode.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.gemma4_z64_hpc import exact_mcnemar
from experiments.r061_qwen38_m1 import final_map_perfect

METRICS = {
    "final_map": final_map_perfect,
    "error_free": lambda record: record["exact"],
}


def compare(before: Path, after: Path, metric: str = "final_map") -> dict:
    score = METRICS[metric]
    left = json.loads(before.read_text())
    right = json.loads(after.read_text())
    def indexed(payload):
        rows = payload["records"]
        indexed = {r["target_id"]: r for r in rows}
        if len(indexed) != len(rows):
            raise ValueError("Duplicate target IDs")
        return indexed
    a, b = indexed(left), indexed(right)
    if set(a) != set(b) or not a:
        raise ValueError("Paired results require the same nonempty target set")
    ids = sorted(a)
    for key in ids:
        for field in ("target", "seed", "initialization_sha256"):
            if field not in a[key] or field not in b[key] or a[key][field] != b[key][field]:
                raise ValueError(f"Pairing mismatch for {key}: {field}")
        if type(a[key].get("exact")) is not bool or type(b[key].get("exact")) is not bool:
            raise ValueError(f"Missing or invalid exact-success observation for {key}")
        if metric == "final_map" and any(
            not isinstance(row.get("hard_constraints"), dict) or "e_inf" not in row
            for row in (a[key], b[key])
        ):
            raise ValueError(f"Missing final-map observation for {key}")
    x = np.array([score(a[i]) for i in ids], dtype=int)
    y = np.array([score(b[i]) for i in ids], dtype=int)
    wins, losses = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum())
    rng = np.random.default_rng(20260831)
    delta = y - x
    # Bounded batches preserve the same sequential RNG draw stream.
    samples = np.concatenate([delta[rng.integers(0, len(ids), size=(1000, len(ids)))].mean(axis=1)
                              for _ in range(200)])
    return {"metric": metric, "pairs": len(ids), "before_exact": int(x.sum()), "after_exact": int(y.sum()),
            "wins": wins, "losses": losses, "difference_percentage_points": 100 * float(delta.mean()),
            "paired_bootstrap_95_percent_interval": (100 * np.quantile(samples, [0.025, 0.975])).tolist(),
            "bootstrap_resamples": 200000, "bootstrap_seed": 20260831,
            "mcnemar_two_sided_exact_p": exact_mcnemar(wins, losses)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=sorted(METRICS), default="final_map")
    args = parser.parse_args()
    result = compare(args.before, args.after, args.metric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
