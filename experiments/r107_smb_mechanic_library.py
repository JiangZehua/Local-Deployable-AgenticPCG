"""R107 dev-derived verified mechanic library for dynamic SMB control.

The library intentionally keeps the dynamic action representation small:
sixteen joint ``(enemies, jumps)`` backbones and one ordered list of observed
trajectory coin slots per backbone.  It does not store the 96 target-specific
levels in ``F_SMB-v0`` and it never calls the evaluator inside the tool.  The
ordinary optimizer step remains the sole online A* verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.r061_qwen38_m1 import _write_json
from experiments.r080_smb_astar_probe import TARGET_FIRST_ACCEPTANCE, _config
from experiments.r083_smb_atlas import ArchiveEntry, _trajectory_coin_sites
from experiments.r083_smb_baseline import (
    DEFAULT_MANIFEST,
    build_cases,
    preflight as baseline_preflight,
    run as baseline_run,
)
from experiments.r085_smb_spawn_fate import SpawnFateSmbOptimizer
from toolusepcg.level import Level, SMBTiles
from toolusepcg.tools.registry import EditTool, ToolResult


RUN_ID = "R107-UNIFIED-MECHANIC-LIBRARY"
SEED = 107100
TOOL_NAME = "build_smb_from_mechanic_library"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEV_WITNESS_ROOT = (
    REPO_ROOT
    / "benchmarks/smb/backbones"
)
SOURCE_ATLAS_PATH = DEV_WITNESS_ROOT.parent / "backbone_sources.json"

# Every source_case_id is in the frozen dev manifest.  A source row may be a
# coin overlay; backbone_sha256 is its explicit structural parent after coins
# are removed.  Only the first five ordered trajectory sites are needed for
# the declared coin range 0..5.
BACKBONES: dict[str, dict[str, Any]] = {
    "k0_j1": {
        "behavior": (0, 1),
        "source_case_id": "smb-dev-00-k0-c0-j1",
        "source_level_sha256": "55e411100370d24e48f257aa85f2792fb7c5d3c472cd5cb5b589a21fa690129c",
        "backbone_sha256": "55e411100370d24e48f257aa85f2792fb7c5d3c472cd5cb5b589a21fa690129c",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 14), (17, 14)),
    },
    "k0_j2": {
        "behavior": (0, 2),
        "source_case_id": "smb-dev-01-k0-c1-j2",
        "source_level_sha256": "8cc8fbeff6e02e5d54922ce989db81c4c0ef988a8313d028937492e833e3a3e9",
        "backbone_sha256": "1834fbd004b5125b22abf22bf27ee6acd9d7d2145507670bb8f27d29270669fa",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (17, 14)),
    },
    "k0_j3": {
        "behavior": (0, 3),
        "source_case_id": "smb-dev-02-k0-c2-j3",
        "source_level_sha256": "a742fb745b228972cfca77074f70df99cf8567336c39931eb9502b3f8fbec1cc",
        "backbone_sha256": "18b93e53301cbc6927ad025911756ac73d00abc1dbd5c235edfe0ef78e669673",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (17, 14)),
    },
    "k0_j4": {
        "behavior": (0, 4),
        "source_case_id": "smb-dev-03-k0-c3-j4",
        "source_level_sha256": "2c53939e369848909deb90857a2a972856eecd7969401e9a0db2c90236cb477b",
        "backbone_sha256": "44c6cd310347cb28b2feda8bce36610b5874e9d1821172ecdcccdcbc20c042c5",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (17, 14)),
    },
    "k0_j5": {
        "behavior": (0, 5),
        "source_case_id": "smb-dev-05-k0-c4-j5",
        "source_level_sha256": "e17cd78d6a6084beab441b50669d41d369a49fe4a46fb041572e5466b9cca1e5",
        "backbone_sha256": "7185bd9516edb5030557831a33bc76c9f2c3648e4f69807062f8e4c9ebcee491",
        "slots": ((2, 14), (5, 14), (9, 11), (13, 11), (17, 14)),
    },
    "k1_j0": {
        "behavior": (1, 0),
        "source_case_id": "smb-dev-07-k1-c0-j0",
        "source_level_sha256": "16c0411fd8f90f4fa3abe9ad77e3a3ee2a4e83a5910df033a537d98ad12ac62f",
        "backbone_sha256": "16c0411fd8f90f4fa3abe9ad77e3a3ee2a4e83a5910df033a537d98ad12ac62f",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 14), (17, 14)),
    },
    "k1_j1": {
        "behavior": (1, 1),
        "source_case_id": "smb-dev-08-k1-c0-j1",
        "source_level_sha256": "2b72f0c16494103a36f4ff27297328f72c6830330c3c12f456c99896203fc7b2",
        "backbone_sha256": "2b72f0c16494103a36f4ff27297328f72c6830330c3c12f456c99896203fc7b2",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 14), (17, 13)),
    },
    "k1_j2": {
        "behavior": (1, 2),
        "source_case_id": "smb-dev-10-k1-c1-j2",
        "source_level_sha256": "00c3316b0702b19acbcf4e37594fc8602fd0c27c518730e3cdc9730a250d46d6",
        "backbone_sha256": "4eab99fe3139daa9a2ebc0837882939ab83bb8a69fc2b0a213d3f512373eff36",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 11), (16, 11)),
    },
    "k1_j3": {
        "behavior": (1, 3),
        "source_case_id": "smb-dev-12-k1-c2-j3",
        "source_level_sha256": "5d8b9129e33e06d908ead662d8c805c9165ba5925ec55e2f8c0fead91766faff",
        "backbone_sha256": "61ac5568b06b58476f104fc0141a1fac7fd8f34d8336ec53113416bdd38969b8",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (16, 14)),
    },
    "k1_j4": {
        "behavior": (1, 4),
        "source_case_id": "smb-dev-14-k1-c3-j4",
        "source_level_sha256": "85169380a2afc696fad9eefdd339ac7fd973293316ce107f8552415122016748",
        "backbone_sha256": "532a2a69195f81f9002f1df6383774a565e82578973facd1161888f73bea30e0",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (17, 15)),
    },
    "k2_j0": {
        "behavior": (2, 0),
        "source_case_id": "smb-dev-22-k2-c4-j0",
        "source_level_sha256": "0c92ad263332363a27102a0c3af860c79adf1765352d6b190d18b8ea26e765b0",
        "backbone_sha256": "13f8e256a99a04d77f5e5d35a3b7672c5a084e5238c855f9dfd0875d194c57fb",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 14), (17, 13)),
    },
    "k2_j1": {
        "behavior": (2, 1),
        "source_case_id": "smb-dev-17-k2-c0-j1",
        "source_level_sha256": "5bec72bdadb2162129a839332190c4663362f1c66966da8de7281df0dfaa8172",
        "backbone_sha256": "5bec72bdadb2162129a839332190c4663362f1c66966da8de7281df0dfaa8172",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 14), (16, 14)),
    },
    "k2_j2": {
        "behavior": (2, 2),
        "source_case_id": "smb-dev-18-k2-c0-j2",
        "source_level_sha256": "0d5f0b73477655d665e132c0e25373230e8c361d311f8b2b998bd05ed0742c49",
        "backbone_sha256": "0d5f0b73477655d665e132c0e25373230e8c361d311f8b2b998bd05ed0742c49",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 11), (16, 11)),
    },
    "k2_j3": {
        "behavior": (2, 3),
        "source_case_id": "smb-dev-20-k2-c1-j3",
        "source_level_sha256": "379fcec0389016c9908d6c9948f57d59de6ad6d1f582851d00952de1a3dfcaa2",
        "backbone_sha256": "d4d6e9100e4ff84342576bae0762d852ab98d3c2fc04cf3d8c711bb323332d30",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (16, 14)),
    },
    "k3_j0": {
        "behavior": (3, 0),
        "source_case_id": "smb-dev-25-k3-c2-j0",
        "source_level_sha256": "fba77ffb98acb14343b3f62caa3d9ddfb8bc6e6f00c3df4f95762262ff7cffe7",
        "backbone_sha256": "5cd951f3169c4a2774aa303b4983998e4ff0ca3967f150b180c13b2b3d70df00",
        "slots": ((2, 14), (5, 14), (9, 14), (13, 15), (17, 14)),
    },
    "k3_j1": {
        "behavior": (3, 1),
        "source_case_id": "smb-dev-27-k3-c3-j1",
        "source_level_sha256": "2fa993b810bd1719861552d222bc38ead411869e069d6dc49f57263a20a88474",
        "backbone_sha256": "10d1bcf152e26f444716f605474a5fcebec8888258f17b2b1bc931b8b86c5dac",
        "slots": ((2, 14), (5, 14), (9, 10), (13, 13), (17, 13)),
    },
}
BACKBONE_IDS = tuple(BACKBONES)


def _catalog_payload() -> dict[str, Any]:
    return {
        key: {
            "behavior": list(value["behavior"]),
            "source_case_id": value["source_case_id"],
            "source_level_sha256": value["source_level_sha256"],
            "backbone_sha256": value["backbone_sha256"],
            "slots": [list(site) for site in value["slots"]],
        }
        for key, value in BACKBONES.items()
    }


LIBRARY_SHA256 = hashlib.sha256(
    json.dumps(_catalog_payload(), sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


def _level_sha256(level: Level) -> str:
    return hashlib.sha256(level.to_string().encode()).hexdigest()


def load_backbone(backbone_id: str) -> Level:
    profile = BACKBONES[backbone_id]
    # Load only the explicitly named DEV target witness, then invert the
    # trajectory-coin overlay.  This avoids dereferencing a held-out target or
    # a target-specific exact level at runtime.
    source_digest = str(profile["source_level_sha256"])
    path = DEV_WITNESS_ROOT / f"{source_digest}.txt"
    if not path.is_file():
        raise RuntimeError(f"missing frozen dev witness: {path}")
    text = path.read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != source_digest:
        raise RuntimeError(f"dev source file hash drift: {backbone_id}")
    level = Level.from_string(text, problem_type="smb")
    level.data[level.data == int(SMBTiles.COIN)] = int(SMBTiles.EMPTY)
    if _level_sha256(level) != profile["backbone_sha256"]:
        raise RuntimeError(f"coin-stripped backbone hash drift: {backbone_id}")
    return level


def intended_parameters(targets: dict[str, Any]) -> dict[str, Any]:
    enemies = int(targets["enemies"])
    coins = int(targets["coins"])
    jumps = int(targets["jumps"])
    return {
        "backbone_id": f"k{enemies}_j{jumps}",
        "coin_slot_ids": list(range(coins)),
    }


class BuildSmbFromMechanicLibraryTool(EditTool):
    """Atomically instantiate one joint mechanic backbone plus coin slots."""

    @property
    def name(self) -> str:
        return TOOL_NAME

    @property
    def description(self) -> str:
        catalog = ", ".join(BACKBONE_IDS)
        return (
            "Rebuild a dynamic 16x32 SMB level from one dev-derived verified "
            "joint kill/jump backbone and explicit observed-trajectory coin "
            f"slots. Available backbone IDs are: {catalog}. ID kN_jM has N "
            "enemy kills and M ground jumps. For C collected coins, pass the "
            "exact ordered prefix coin_slot_ids=[0,...,C-1]; use [] for zero. "
            "The tool does not run A*: the ordinary STEP evaluator verifies "
            "the resulting gameplay once."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "backbone_id": {
                    "type": "string",
                    "enum": list(BACKBONE_IDS),
                    "description": "Joint kill/jump mechanic profile kN_jM.",
                },
                "coin_slot_ids": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 4},
                    "minItems": 0,
                    "maxItems": 5,
                    "description": "Ordered prefix [0,...,C-1] for C coins.",
                },
            },
            "required": ["backbone_id", "coin_slot_ids"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        backbone_id = str(kwargs.get("backbone_id", ""))
        if backbone_id not in BACKBONES:
            return ToolResult(
                success=False,
                errors=[f"backbone_id must be one of {list(BACKBONE_IDS)}"],
            )
        raw_slots = kwargs.get("coin_slot_ids")
        if not isinstance(raw_slots, list) or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_slots
        ):
            return ToolResult(
                success=False,
                errors=["coin_slot_ids must be an array of integer slot IDs"],
            )
        if len(raw_slots) > 5 or raw_slots != list(range(len(raw_slots))):
            return ToolResult(
                success=False,
                errors=[
                    "coin_slot_ids must be the unique ordered prefix "
                    "[0,...,C-1] with 0 <= C <= 5"
                ],
            )
        manager = self._edit_manager
        if manager.problem_type != "smb" or (manager.height, manager.width) != (16, 32):
            return ToolResult(
                success=False,
                errors=["mechanic library requires a dynamic 16x32 SMB level"],
            )

        profile = BACKBONES[backbone_id]
        target = load_backbone(backbone_id).data.copy()
        selected_sites = []
        for slot_id in raw_slots:
            x, y = profile["slots"][slot_id]
            if int(target[y, x]) != int(SMBTiles.EMPTY):
                return ToolResult(
                    success=False,
                    errors=[f"certified coin slot {slot_id} is no longer empty"],
                )
            target[y, x] = int(SMBTiles.COIN)
            selected_sites.append([int(x), int(y)])

        before = manager.level_data
        changed = np.argwhere(before != target)
        checkpoint = manager.checkpoint()
        try:
            manager.set_level(Level(target, problem_type="smb"))
        except Exception as exc:
            manager.rollback(checkpoint)
            return ToolResult(
                success=False,
                errors=[f"transaction rolled back: {type(exc).__name__}: {exc}"],
            )
        enemies, jumps = profile["behavior"]
        return ToolResult(
            success=True,
            data={
                "new_level": manager.level.to_string(),
                "applied": True,
                "transactional": True,
                "operator_contract": "r107-dev-mechanic-library-v1",
                "library_sha256": LIBRARY_SHA256,
                "backbone_id": backbone_id,
                "source_case_id": profile["source_case_id"],
                "backbone_sha256": profile["backbone_sha256"],
                "coin_slot_ids": list(raw_slots),
                "coin_sites": selected_sites,
                "intended_behavior": {
                    "enemies": int(enemies),
                    "coins": len(raw_slots),
                    "jumps": int(jumps),
                },
                "num_tiles_changed": int(len(changed)),
            },
        )


def _actual_behavior(result: Any) -> tuple[int, int, int] | None:
    try:
        if not result.valid or float(result.metrics.get("complete", 0.0)) != 1.0:
            return None
        return tuple(int(result.metrics[name]) for name in ("enemies", "coins", "jumps"))
    except (KeyError, TypeError, ValueError):
        return None


class MechanicLibrarySmbOptimizer(SpawnFateSmbOptimizer):
    def initialize(self, seed: int | None = None, initial_level: Level | None = None):
        super().initialize(seed=seed, initial_level=initial_level)
        if self.registry.get(TOOL_NAME) is None:
            self.registry.register(BuildSmbFromMechanicLibraryTool(self.edit_manager))
        self._openai_structured_outputs = None
        self._openai_step_schema = None

    def execute_step(self, message):
        library_calls = [
            call for call in message.tool_calls if call.tool_name == TOOL_NAME
        ]
        if library_calls and len(message.tool_calls) != 1:
            return (
                self.current_level.copy(),
                [{
                    "tool_name": TOOL_NAME,
                    "parameters": library_calls[0].parameters,
                    "result": ToolResult(
                        success=False,
                        errors=["mechanic-library rebuild must be the only call"],
                    ).to_dict(),
                }],
                0,
            )
        level, results, changed = super().execute_step(message)
        changed += sum(
            int((item.get("result") or {}).get("data", {}).get("num_tiles_changed", 0))
            for item in results
            if item.get("tool_name") == TOOL_NAME
            and (item.get("result") or {}).get("success")
        )
        return level, results, changed

    def _build_environment_step_feedback(self, new_level, new_result, tool_results):
        base = super()._build_environment_step_feedback(
            new_level, new_result, tool_results
        )
        calls = [item for item in tool_results if item.get("tool_name") == TOOL_NAME]
        if not calls:
            return base
        data = (calls[0].get("result") or {}).get("data") or {}
        intended = data.get("intended_behavior")
        return "\n".join(
            (
                base,
                "Mechanic-library verifier (candidate-local):",
                f"- declared={intended}; actual={_actual_behavior(new_result)}; "
                f"target={tuple(int(getattr(self.config.targets, name).value) for name in ('enemies', 'coins', 'jumps'))}",
                "- If non-exact, correct the backbone ID or ordered coin-slot prefix; do not invent raw tile edits.",
            )
        )


LIBRARY_INSTRUCTION = (
    "Use the verified SMB mechanic library, not raw tile edits. In the first "
    "STEP call build_smb_from_mechanic_library exactly once. For targets "
    "enemies=K and jumps=J choose backbone_id='kK_jJ'. For coins=C choose "
    "coin_slot_ids=[0,1,...,C-1], or [] when C=0. Example: K=2, C=4, "
    "J=3 uses backbone_id='k2_j3' and coin_slot_ids=[0,1,2,3]. The ordinary "
    "A* result is authoritative; retry only if the observed tuple is not exact."
)


def build_library_optimizer(
    config: Any,
    *,
    acceptance_policy: str = TARGET_FIRST_ACCEPTANCE,
) -> MechanicLibrarySmbOptimizer:
    if acceptance_policy != TARGET_FIRST_ACCEPTANCE:
        raise ValueError("R107 requires target-first acceptance")
    target_id = f"k{int(config.targets.enemies.value)}_j{int(config.targets.jumps.value)}"
    if target_id not in BACKBONES:
        raise ValueError(f"R107 unsupported joint mechanic target: {target_id}")
    config.optimizer.edit_tools = []
    config.optimizer.max_tool_calls_per_step = 1
    config.optimizer.stable_change_accounting = True
    base = config.optimizer.extra_instruction
    config.optimizer.extra_instruction = (
        f"{base}\n\n{LIBRARY_INSTRUCTION}" if base else LIBRARY_INSTRUCTION
    )
    optimizer = MechanicLibrarySmbOptimizer(config, verbose=False)
    optimizer.duplicate_level_guard = True
    return optimizer


def _target_tuple(targets: dict[str, Any]) -> tuple[int, int, int]:
    return tuple(int(targets[name]) for name in ("enemies", "coins", "jumps"))


def _hard_valid(result: Any) -> bool:
    info = result.info or {}
    return bool(
        result.valid
        and float(result.metrics.get("complete", 0.0)) == 1.0
        and info.get("engine_executed")
        and info.get("actions")
    )


def preflight(
    manifest: Path,
    output: Path,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    payload = baseline_preflight(
        manifest,
        output,
        seed=seed,
        optimizer_builder=build_library_optimizer,
        metadata={
            "experiment": RUN_ID,
            "library_sha256": LIBRARY_SHA256,
            "target_specific_oracle_lookup": False,
            "online_hidden_evaluator_calls": 0,
        },
    )
    payload["oracle_witnesses_loaded"] = True
    payload["oracle_scope"] = "16 dev-derived joint-mechanic backbone ancestors"
    payload["validation_or_sealed_witnesses_loaded"] = False
    payload["target_specific_exact_witness_loaded"] = False
    _, cases = build_cases(manifest, seed=seed)
    case_by_id = {case["case_id"]: case for case in cases}
    source_atlas_bytes = SOURCE_ATLAS_PATH.read_bytes()
    source_atlas = json.loads(source_atlas_bytes)
    certified_atlas_rows = {
        _target_tuple(row["target"]): row
        for row in source_atlas["expanded_manifest"]
        if row.get("status") == "certified_feasible"
    }
    optimizer = build_library_optimizer(
        _config(cases[0], model_id="preflight", base_url="http://127.0.0.1:1/v1")
    )
    optimizer.initialize(seed=seed)
    schema = optimizer._build_openai_extra_body()["structured_outputs"]["json"]
    decoded = {
        branch["properties"]["tool_name"]["const"]
        for branch in schema["properties"]["tool_calls"]["items"]["oneOf"]
    }
    if decoded != {TOOL_NAME} or decoded != set(optimizer.registry.list_tools()):
        raise RuntimeError("R107 runtime registry and decoding schema diverged")

    certified = []
    for backbone_id, profile in BACKBONES.items():
        source = case_by_id.get(profile["source_case_id"])
        if source is None:
            raise RuntimeError(f"non-dev source case in R107 library: {backbone_id}")
        source_target = _target_tuple(source["targets"])
        if (source_target[0], source_target[2]) != tuple(profile["behavior"]):
            raise RuntimeError(f"source behavior mismatch: {backbone_id}")
        atlas_row = certified_atlas_rows.get(source_target)
        if atlas_row is None or atlas_row.get("level_sha256") != profile["source_level_sha256"]:
            raise RuntimeError(f"dev source is not the certified atlas row: {backbone_id}")
        level = load_backbone(backbone_id)
        result = optimizer.evaluator.evaluate(level)
        expected = (profile["behavior"][0], 0, profile["behavior"][1])
        if not _hard_valid(result) or _actual_behavior(result) != expected:
            raise RuntimeError(f"backbone verification failed: {backbone_id}")
        derived_slots = _trajectory_coin_sites(
            ArchiveEntry(
                behavior=expected,
                level=level,
                family="r107-preflight",
                parameters={},
                primitive_cost=0,
                level_sha256=profile["backbone_sha256"],
                result=result,
            )
        )[:5]
        if tuple(derived_slots) != tuple(profile["slots"]):
            raise RuntimeError(f"trajectory coin-slot derivation drift: {backbone_id}")
        certified.append(
            {
                "backbone_id": backbone_id,
                "source_case_id": profile["source_case_id"],
                "source_level_sha256": profile["source_level_sha256"],
                "backbone_sha256": _level_sha256(level),
                "behavior": list(expected),
                "trajectory_coin_slots": [list(site) for site in profile["slots"]],
            }
        )
    payload["r107_library_contract"] = {
        "gate_pass": len(certified) == 16,
        "dev_derived_backbones": len(certified),
        "joint_mechanic_profiles": sorted(BACKBONE_IDS),
        "coin_rule": "ordered observed-trajectory prefix, length 0..5",
        "source_atlas_sha256": hashlib.sha256(source_atlas_bytes).hexdigest(),
        "source_rows_checked_against_atlas": len(certified),
        "coin_slots_rederived_from_backbone_trajectory": len(certified),
        "transactional_tool": TOOL_NAME,
        "ordinary_candidate_evaluator_only": True,
        "schema_tools": sorted(decoded),
        "certified": certified,
    }
    _write_json(output, payload)
    return payload


def deterministic_ceiling(
    manifest: Path,
    output: Path,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    manifest_payload, cases = build_cases(manifest, seed=seed)
    optimizer = build_library_optimizer(
        _config(cases[0], model_id="no-llm", base_url="http://127.0.0.1:1/v1")
    )
    optimizer.initialize(seed=seed)
    tool = optimizer.registry.get(TOOL_NAME)
    assert isinstance(tool, BuildSmbFromMechanicLibraryTool)
    rows = []
    for case in cases:
        parameters = intended_parameters(case["targets"])
        tool_result = tool.execute(**parameters)
        result = optimizer.evaluator.evaluate(optimizer.edit_manager.level)
        actual = _actual_behavior(result)
        target = _target_tuple(case["targets"])
        rows.append(
            {
                "case_id": case["case_id"],
                "target": list(target),
                "parameters": parameters,
                "tool_success": tool_result.success,
                "hard_valid": _hard_valid(result),
                "actual": list(actual) if actual is not None else None,
                "exact": bool(tool_result.success and _hard_valid(result) and actual == target),
                "level_sha256": _level_sha256(optimizer.edit_manager.level),
            }
        )
    exact = sum(row["exact"] for row in rows)
    payload = {
        "run_id": RUN_ID,
        "arm": "deterministic-no-llm-library-ceiling",
        "manifest_semantic_sha256": manifest_payload["semantic_sha256"],
        "library_sha256": LIBRARY_SHA256,
        "cases": len(rows),
        "candidate_astar_evaluations": len(rows),
        "initialization_astar_evaluations": 1,
        "exact": exact,
        "hard_valid": sum(row["hard_valid"] for row in rows),
        "gate_pass": exact == len(rows) and all(row["hard_valid"] for row in rows),
        "records": rows,
    }
    _write_json(output, payload)
    return payload


def _tool_calls(record: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    calls = []
    for step in record["steps"]:
        for call in (step.get("final_json") or {}).get("tool_calls", []):
            if call.get("tool_name") == TOOL_NAME:
                calls.append((call, step))
    return calls


def _audit_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    first_choice_correct = 0
    first_candidate_exact = 0
    total_calls = 0
    cases_with_call = 0
    per_case = []
    for record in records:
        expected = intended_parameters(record["targets"])
        calls = _tool_calls(record)
        total_calls += len(calls)
        cases_with_call += int(bool(calls))
        first_params = calls[0][0].get("parameters") if calls else None
        correct = first_params == expected
        first_choice_correct += int(correct)
        metrics = calls[0][1].get("metrics_after", {}) if calls else {}
        actual = tuple(int(metrics.get(name, -1)) for name in ("enemies", "coins", "jumps"))
        exact = bool(calls and float(metrics.get("complete", 0.0)) == 1.0 and actual == _target_tuple(record["targets"]))
        first_candidate_exact += int(exact)
        per_case.append(
            {
                "case_id": record["case_id"],
                "expected_parameters": expected,
                "first_parameters": first_params,
                "first_choice_correct": correct,
                "first_candidate_exact": exact,
                "library_calls": len(calls),
            }
        )
    return {
        "library_calls": total_calls,
        "cases_with_library_call": cases_with_call,
        "first_choice_correct": first_choice_correct,
        "first_candidate_exact": first_candidate_exact,
        "max_library_calls_per_case": max((row["library_calls"] for row in per_case), default=0),
        "per_case": per_case,
    }


def run(
    manifest: Path,
    base_url: str,
    output: Path,
    *,
    model_id: str,
    revision: str,
    max_steps: int = 4,
    seed: int = SEED,
) -> dict[str, Any]:
    payload = baseline_run(
        manifest,
        base_url,
        output,
        model_id=model_id,
        revision=revision,
        max_steps=max_steps,
        max_protocol_retries=4,
        temperature=0.0,
        seed=seed,
        optimizer_builder=build_library_optimizer,
        harness_overrides={
            "baseline": "R084-P3 protocol + R085-F2 compact causal feedback",
            "controller": "single-incumbent target-first",
            "semantic_action": TOOL_NAME,
            "dev_mechanic_library_loaded": True,
            "oracle_witnesses_loaded": True,
            "oracle_scope": "16 dev-derived joint-mechanic backbone ancestors",
            "validation_or_sealed_witnesses_loaded": False,
            "target_specific_oracle_witness_loaded": False,
            "library_sha256": LIBRARY_SHA256,
            "semantic_steps": max_steps,
            "candidate_astar_budget": max_steps,
            "online_hidden_evaluator_calls": 0,
            "single_changed_factor": "dev-derived joint mechanic backbone plus trajectory coin slots",
        },
        experiment_name=RUN_ID,
    )
    audit = _audit_records(payload["records"])
    aggregate = payload["aggregate"]
    total = len(payload["records"])
    criteria = {
        "execution_complete": payload["execution_complete"],
        "contract_gate_pass": payload["contract_gate_pass"],
        "hard_valid_all": aggregate["hard_valid"] == total,
        "no_unrecovered_parser_errors": aggregate["unrecovered_parser_errors"] == 0,
        "no_transport_errors": aggregate["transport_errors"] == 0,
        "no_tool_errors": aggregate["tool_errors"] == 0,
        "library_called_for_every_case": audit["cases_with_library_call"] == total,
        "candidate_budget_bounded": audit["max_library_calls_per_case"] <= max_steps,
        "promotion_exact": (
            audit["first_candidate_exact"] >= 24 or aggregate["exact"] >= 27
        ),
    }
    payload["r107_library_audit"] = audit
    payload["r107_gate"] = {
        "gate_pass": all(criteria.values()),
        "criteria": criteria,
        "classification": (
            "harness-assisted functional control; not raw-model SMB mastery"
        ),
        "historical_context": {
            "R083_D1_temperature0_exact": "3/29",
            "R099_independent_sampling_pass_at_4": "11/29",
            "R100_R101_R102_selected_operator_smokes": "18/18",
        },
    }
    _write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "ceiling", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, default=SEED)
        if name == "run":
            command.add_argument("--base-url", required=True)
            command.add_argument("--model-id", required=True)
            command.add_argument("--revision", required=True)
            command.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args()
    if args.command == "preflight":
        payload = preflight(args.manifest, args.output, seed=args.seed)
    elif args.command == "ceiling":
        payload = deterministic_ceiling(args.manifest, args.output, seed=args.seed)
    else:
        payload = run(
            args.manifest,
            args.base_url,
            args.output,
            model_id=args.model_id,
            revision=args.revision,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
