"""Offline construction, execution and rendering checks; no model calls."""
import argparse
import json
from pathlib import Path

import toolusepcg
from toolusepcg.config import load_config
from toolusepcg.evaluator import StatsEvaluator
from toolusepcg.factory import LevelFactory
from toolusepcg.functional_resolver import FunctionalSpec, MetricConstraint
from toolusepcg.level import Level
from experiments.zelda_mastery_m0 import construct_zelda_corridor
from experiments.smb_dynamic_builders import build_resolver


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config("configs/zelda.yaml")
    factory = LevelFactory(cfg)
    level = construct_zelda_corridor(17, 31)
    result = StatsEvaluator(cfg, factory.env).evaluate(level)
    assert result.valid and result.metrics["player_key"] == 17 and result.metrics["key_door"] == 31
    (output / "zelda.txt").write_text(level.to_string() + "\n")
    factory.env.render(level.data).save(output / "zelda.png")
    resolution = build_resolver("stitch", contract="v2").resolve(FunctionalSpec("smb_dynamic", {
        key: MetricConstraint("exact", value) for key, value in {"enemies": 3, "coins": 5, "jumps": 1}.items()
    }))
    payload = resolution.as_dict()
    assert payload["status"] == "exact", payload
    smb = Level.from_string(payload["level"], "smb")
    (output / "smb.txt").write_text(smb.to_string() + "\n")
    LevelFactory(load_config("configs/smb.yaml")).env._problem.render(smb.data).save(output / "smb.png")
    report = {"offline": True, "model_calls": 0, "zelda": {"player_key": 17, "key_door": 31},
              "smb": {"enemies": 3, "coins": 5, "jumps": 1, "contract": "v2",
                      "takeoff_audit": resolution.evaluation_info["takeoff_audit"]}, "passed": True}
    (output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
