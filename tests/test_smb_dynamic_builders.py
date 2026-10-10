from __future__ import annotations

import random
import unittest

import numpy as np

from experiments.smb_dynamic_builders import (
    DYNAMIC_NAMES,
    PatchLibrary,
    SmbDynamicBuilderAdapter,
    build_resolver,
    envelope_targets,
    mutate,
    random_patch,
    stitch_plans,
    target_seed,
)
from toolusepcg.functional_resolver import FunctionalObjective, FunctionalSpec, MetricConstraint
from toolusepcg.level import SMBTiles
from toolusepcg.smb_patches import FAMILIES, MIN_WIDTH, Patch, assemble


def _dynamic(enemies: int, coins: int, jumps: int) -> FunctionalSpec:
    return FunctionalSpec(
        "smb_dynamic",
        {
            "enemies": MetricConstraint("exact", enemies),
            "coins": MetricConstraint("exact", coins),
            "jumps": MetricConstraint("exact", jumps),
        },
    )


class PatchContractTests(unittest.TestCase):
    def test_every_family_keeps_floor_at_both_edges(self) -> None:
        rng = random.Random(0)
        for name, family in FAMILIES.items():
            for _ in range(20):
                while True:
                    params = {k: rng.randint(lo, hi) for k, (lo, hi) in family.ranges.items()}
                    if family.valid_params(params):
                        break
                patch = Patch.make(name, **params)
                data = patch.render()
                self.assertEqual(data.shape[0], 16)
                from toolusepcg.smb_patches import FAMILY_SETS
                floor = 14 if name in FAMILY_SETS["v1"] else 15  # v2: one-row floor
                for column in (0, data.shape[1] - 1):
                    self.assertTrue(np.all(data[floor:, column] == int(SMBTiles.SOLID)), patch.token())
                if name in FAMILY_SETS["v2"]:
                    self.assertFalse(np.any(data[:13] == int(SMBTiles.SOLID)) and name != "stomp_pillar"
                                     and name != "obstacle15", f"v2 patch has a ceiling: {patch.token()}")

    def test_assemble_left_pads_corridor_to_min_width(self) -> None:
        level = assemble([Patch.make("obstacle", width=8, height=1, x=7)])
        self.assertEqual(level.width, MIN_WIDTH)
        self.assertEqual(int(level.data[12, 0]), int(SMBTiles.SOLID))
        self.assertEqual(int(level.data[13, MIN_WIDTH - 1]), int(SMBTiles.SOLID))

    def test_patch_validation_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            Patch.make("stomp", width=10, enemy_x=9)
        with self.assertRaises(ValueError):
            Patch.make("gap", width=6, gap=4, x=3)
        patch = Patch.make("coins", count=3, spacing=2)
        self.assertEqual(Patch.from_dict(patch.as_dict()), patch)


class StitchPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library = PatchLibrary.load()

    def test_every_envelope_target_has_a_predicting_plan(self) -> None:
        stomps = set(map(Patch.token, self.library.roles["stomp"]))
        jumps = set(map(Patch.token, self.library.roles["jump"]))
        exits = set(map(Patch.token, self.library.roles["exit"]))
        self.assertEqual(len(envelope_targets()), 210)
        for target in envelope_targets():
            plans = list(stitch_plans(self.library, target))
            self.assertTrue(plans, target)
            for plan, meta in plans:
                tokens = [p.token() for p in plan]
                self.assertEqual(sum(t in stomps for t in tokens), target["enemies"])
                coin_patches = [p for p in plan if p.family == "coins"]
                self.assertEqual(sum(p.p["count"] for p in coin_patches), target["coins"])
                predicted_jumps = sum(t in jumps for t in tokens[:-1]) + (tokens[-1] in exits)
                self.assertEqual(predicted_jumps, target["jumps"])
                self.assertEqual(meta["coin_position"] is None, target["coins"] == 0)

    def test_plan_order_is_deterministic_and_unique(self) -> None:
        target = {"enemies": 2, "coins": 3, "jumps": 4}
        first = [m["plan_sha256"] for _, m in stitch_plans(self.library, target)]
        second = [m["plan_sha256"] for _, m in stitch_plans(self.library, target)]
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))


class SearchTests(unittest.TestCase):
    def test_target_seed_and_mutation_are_deterministic(self) -> None:
        target = {"enemies": 1, "coins": 2, "jumps": 3}
        self.assertEqual(target_seed(target, 1), target_seed(dict(target), 1))
        self.assertNotEqual(target_seed(target, 1), target_seed(target, 2))
        a = mutate(random.Random(5), [], {"enemies": 0, "coins": 0, "jumps": 0}, target)
        b = mutate(random.Random(5), [], {"enemies": 0, "coins": 0, "jumps": 0}, target)
        self.assertEqual(a, b)

    def test_directed_insert_uses_role_of_missing_metric(self) -> None:
        rng = random.Random(3)
        target = {"enemies": 2, "coins": 0, "jumps": 0}
        roles = set()
        for _ in range(200):
            child = mutate(rng, [], {"enemies": 0, "coins": 0, "jumps": 0}, target)
            roles.update(p.role for p in child)
        self.assertIn("enemies", roles)


class AdapterTests(unittest.TestCase):
    def test_validation_matches_legacy_envelope(self) -> None:
        adapter = SmbDynamicBuilderAdapter("search")
        self.assertEqual(adapter.validate(_dynamic(4, 5, 6)), [])
        self.assertTrue(adapter.validate(_dynamic(5, 0, 0)))
        foreign = FunctionalSpec("smb_dynamic", {"pipe_count": MetricConstraint("exact", 1)})
        self.assertTrue(adapter.validate(foreign))

    def test_range_spec_selects_envelope_target(self) -> None:
        adapter = SmbDynamicBuilderAdapter("search")
        spec = FunctionalSpec("smb_dynamic", {"coins": MetricConstraint("min", 2)},
                              (FunctionalObjective("enemies", "maximize"),))
        self.assertEqual(adapter.select_target(spec), {"enemies": 4, "coins": 2, "jumps": 0})

    def test_stitch_resolves_beyond_legacy_library(self) -> None:
        resolver = build_resolver("stitch")
        for target in ((0, 0, 0), (4, 1, 2)):
            resolution = resolver.resolve(_dynamic(*target), max_evaluator_calls=16)
            self.assertEqual(resolution.status, "exact", target)
            self.assertEqual(resolution.strategy, "certified_patch_stitching")
            observed = tuple(int(resolution.observed_metrics[n]) for n in DYNAMIC_NAMES)
            self.assertEqual(observed, target)
            self.assertGreaterEqual(resolution.level.width, MIN_WIDTH)


if __name__ == "__main__":
    unittest.main()


class ContractV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from experiments.smb_dynamic_builders import PatchLibraryV2

        cls.library = PatchLibraryV2.load()

    def test_v2_plans_predict_every_target_except_k0_j0(self) -> None:
        from experiments.smb_dynamic_builders import stitch_plans_v2

        stomp_sizes = {p.token(): n for role in ("stomp_mid", "stomp_end")
                       for n, ps in self.library.stomps[role].items() for p in ps}
        for target in envelope_targets():
            plans = list(stitch_plans_v2(self.library, target))
            if target["enemies"] == 0 and target["jumps"] == 0:
                self.assertEqual(plans, [], target)
                continue
            self.assertTrue(plans, target)
            for plan, meta in plans:
                self.assertTrue(all(p.family in ("flat", "obstacle15", "gap15", "coins15", "stomp_cluster")
                                    for p in plan), "v2 plans must be ceiling-free")
                self.assertEqual(sum(stomp_sizes.get(p.token(), 0) for p in plan
                                     if p.family == "stomp_cluster"), target["enemies"])
                self.assertEqual(sum(p.p["count"] for p in plan if p.family == "coins15"), target["coins"])

    def test_v1_search_samples_only_v1_families(self) -> None:
        from toolusepcg.smb_patches import FAMILY_SETS

        rng = random.Random(0)
        for _ in range(300):
            self.assertIn(random_patch(rng).family, FAMILY_SETS["v1"])
            self.assertIn(random_patch(rng, None, "v2").family, FAMILY_SETS["v2"])

    def test_takeoff_audit_flags_corridor_stomp(self) -> None:
        from toolusepcg.smb_patches import WidthEvaluatorPool

        pool = WidthEvaluatorPool()
        corridor = assemble([Patch.make("corridor", width=4), Patch.make("stomp", width=10, enemy_x=6),
                             Patch.make("corridor", width=4)])
        audit = pool.evaluate_audited(corridor).info["takeoff_audit"]
        self.assertGreater(audit["hidden_takeoffs"], 0)
        flat = assemble([Patch.make("flat", width=8)], pad="flat")
        audit = pool.evaluate_audited(flat).info["takeoff_audit"]
        self.assertEqual(audit["hidden_takeoffs"], 0)
        self.assertEqual(audit["takeoffs"], audit["jump_events"])

    def test_v2_stitch_resolves_with_clean_takeoffs(self) -> None:
        resolver = build_resolver("stitch", contract="v2")
        resolution = resolver.resolve(_dynamic(2, 3, 1), max_evaluator_calls=16)
        self.assertEqual(resolution.status, "exact")
        self.assertEqual(resolver.resolve(_dynamic(0, 0, 0), max_evaluator_calls=16).status, "unknown")
