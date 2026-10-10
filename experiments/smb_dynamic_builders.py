"""Dynamic SMB builders: certified patch stitching and simulation-guided search.

Two construction strategies replace the R107 whole-level lookup:

* ``stitch`` composes a variable-width level from the certified patch library
  (``experiments/smb_patch_library.py``) using a fixed grammar
  ``lead [coins] stomp^k [coins] jump^(j-1) exit|tail``.  Certified per-patch
  event counts make every plan predict the target exactly; the A* verifier
  then checks the assembled level, and a bounded list of alternative plans
  (variant rotation, coin position) is tried in a deterministic order.
* ``search`` runs a (1+1) evolutionary search over sequences of *uncertified*
  patch primitives, guided only by A* feedback (L1 event error), with
  error-directed insert/delete mutations.  It never reads the library.

``stitch_then_search`` falls back to search seeded with the first stitched
plan.  Every reported success is one complete, valid A* playthrough whose
event counts equal the selected target.

Contracts: ``v1`` (PAPER-M8) uses the two-row-floor, low-ceiling patch set.
``v2`` (PAPER-M8b) uses the ceiling-free set and additionally requires every
physical ground takeoff to be a counted JUMP event (zero hidden takeoffs),
with stomp clusters certified separately for mid-level and end positions.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from experiments.smb_patch_library import COIN_ROLES, DEFAULT_OUTPUT, load_library
from toolusepcg.evaluator import EvalResult
from toolusepcg.functional_resolver import (
    FunctionalCandidate,
    FunctionalResolver,
    FunctionalSpec,
    SearchOutcome,
    select_satisfying_target,
)
from toolusepcg.level import Level
from toolusepcg.smb_contract import get_smb_target_envelope
from toolusepcg.smb_patches import (
    FAMILIES,
    FAMILY_SETS,
    Patch,
    WidthEvaluatorPool,
    assemble,
    level_sha256,
    plan_payload,
    plan_sha256,
)


DYNAMIC_NAMES = ("enemies", "coins", "jumps")
MODES = ("stitch", "search", "stitch_then_search")
CONTRACTS = ("v1", "v2")
PAD = {"v1": "corridor", "v2": "flat"}
SEED = 2026100
DEFAULT_STITCH_PLANS = 16
MAX_GENOME = 24
MAX_WIDTH = 192
STAGNATION_RESTART = 32
INVALID_PENALTY = 100


def functional_exact(result: EvalResult, target: Mapping[str, int]) -> bool:
    return bool(
        result.valid
        and float(result.metrics.get("complete", 0.0)) == 1.0
        and all(int(result.metrics.get(name, -1)) == int(value) for name, value in target.items())
    )


def contract_exact(result: EvalResult, target: Mapping[str, int], contract: str) -> bool:
    if not functional_exact(result, target):
        return False
    if contract == "v2":
        from toolusepcg.smb_takeoff_audit import takeoffs_clean

        return takeoffs_clean(result)
    return True


def observed_tuple(result: EvalResult | None) -> dict[str, int] | None:
    if result is None:
        return None
    try:
        return {name: int(result.metrics[name]) for name in DYNAMIC_NAMES}
    except (KeyError, TypeError, ValueError):
        return None


def envelope_targets() -> list[dict[str, int]]:
    envelope = get_smb_target_envelope("expanded")
    return [
        {"enemies": k, "coins": c, "jumps": j}
        for k in range(envelope["enemies"][0], envelope["enemies"][1] + 1)
        for c in range(envelope["coins"][0], envelope["coins"][1] + 1)
        for j in range(envelope["jumps"][0], envelope["jumps"][1] + 1)
    ]


# --------------------------------------------------------------------------
# (a) certified patch stitching


class PatchLibrary:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.sha256 = str(payload["library_sha256"])
        self.lead = Patch.from_dict(payload["lead"])
        certified = payload["certified"]
        self.roles = {
            role: [Patch.from_dict(item) for item in certified[role]]
            for role in ("stomp", "jump", "exit", "sep", "tail")
        }
        self.coins: dict[str, dict[int, list[Patch]]] = {}
        for role in COIN_ROLES:
            by_count: dict[int, list[Patch]] = {}
            for item in certified.get(role, []):
                patch = Patch.from_dict(item)
                by_count.setdefault(patch.p["count"], []).append(patch)
            self.coins[role] = by_count
        for role, patches in self.roles.items():
            if not patches:
                raise RuntimeError(f"patch library has no certified {role} patch")

    @classmethod
    def load(cls, path: Path = DEFAULT_OUTPUT) -> "PatchLibrary":
        return cls(load_library(path))

    def coin_positions(self, count: int) -> list[str]:
        return [role for role in ("coins_after_stomps", "coins_after_lead")
                if self.coins[role].get(count)]


def stitch_plans(library: PatchLibrary, target: Mapping[str, int]) -> Iterator[tuple[list[Patch], dict[str, Any]]]:
    """Deterministic, de-duplicated plans that each predict ``target`` exactly."""
    k, c, j = (int(target[name]) for name in DYNAMIC_NAMES)
    positions: list[str | None] = library.coin_positions(c) if c else [None]
    if c and not positions:
        return
    roles = library.roles
    rotations = max(
        [len(patches) for patches in roles.values()]
        + [len(library.coins[pos][c]) for pos in positions if pos is not None]
    )
    seen: set[str] = set()
    for r in range(rotations):
        pick = {role: patches[r % len(patches)] for role, patches in roles.items()}
        for position in positions:
            if position is not None and k == 0 and position == "coins_after_lead":
                # Identical layout to coins_after_stomps when k == 0.
                continue
            coins = None
            if position is not None:
                variants = library.coins[position][c]
                coins = variants[r % len(variants)]
            sep = pick["sep"]
            plan = [library.lead]
            if position == "coins_after_lead":
                plan += [coins, sep]
            for _ in range(k):
                plan += [pick["stomp"], sep]
            if position == "coins_after_stomps":
                plan += [coins, sep]
            for _ in range(max(j - 1, 0)):
                plan += [pick["jump"], sep]
            plan.append(pick["exit"] if j else pick["tail"])
            digest = plan_sha256(plan)
            if digest in seen:
                continue
            seen.add(digest)
            yield plan, {"rotation": r, "coin_position": position, "plan_sha256": digest}


class PatchLibraryV2:
    """Ceiling-free library: stomp clusters certified per position (mid / end)."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.sha256 = str(payload["library_sha256"])
        self.lead = Patch.from_dict(payload["lead"])
        certified = payload["certified"]
        self.roles = {role: [Patch.from_dict(item) for item in certified[role]]
                      for role in ("jump", "exit", "sep")}
        self.coins: dict[int, list[Patch]] = {}
        for item in certified["coins"]:
            patch = Patch.from_dict(item)
            self.coins.setdefault(patch.p["count"], []).append(patch)
        self.stomps: dict[str, dict[int, list[Patch]]] = {"stomp_mid": {}, "stomp_end": {}}
        for role in self.stomps:
            for item in certified[role]:
                patch = Patch.from_dict(item)
                n = patch.p["count"] if patch.family == "stomp_cluster" else 1
                self.stomps[role].setdefault(n, []).append(patch)
        for role, patches in self.roles.items():
            if not patches:
                raise RuntimeError(f"patch library v2 has no certified {role} patch")

    @classmethod
    def load(cls, path: Path | None = None) -> "PatchLibraryV2":
        from experiments.smb_patch_library_v2 import DEFAULT_OUTPUT as V2_OUTPUT, load_library as load_v2

        return cls(load_v2(path or V2_OUTPUT))


def _partitions(k: int, parts: Sequence[int]) -> list[tuple[int, ...]]:
    """Ordered multisets of available cluster sizes summing to k, fewest clusters first."""
    out: list[tuple[int, ...]] = []

    def rec(rest: int, maximum: int, acc: tuple[int, ...]) -> None:
        if rest == 0:
            out.append(acc)
            return
        for size in sorted((p for p in parts if p <= min(rest, maximum)), reverse=True):
            rec(rest - size, size, acc + (size,))

    rec(k, k, ())
    return sorted(out, key=lambda t: (len(t), [-x for x in t]))


def stitch_plans_v2(library: PatchLibraryV2, target: Mapping[str, int]) -> Iterator[tuple[list[Patch], dict[str, Any]]]:
    """Grammar: lead [coins sep] (stomp_mid sep)* [coins sep] (jump sep)^(j-1) exit
    or, when j == 0, ... stomp_end (the end cluster absorbs the goal jump)."""
    k, c, j = (int(target[name]) for name in DYNAMIC_NAMES)
    mids, ends = library.stomps["stomp_mid"], library.stomps["stomp_end"]
    if j == 0:
        if k == 0:
            return
        options = [(part, end) for end in sorted(ends) if end <= k for part in _partitions(k - end, list(mids))]
    else:
        options = [(part, None) for part in _partitions(k, list(mids))]
    if not options or (c and c not in library.coins):
        return
    positions: list[str | None] = ["coins_after_lead", "coins_after_stomps"] if c else [None]
    if c and k == 0:
        positions = ["coins_after_lead"]
    roles = library.roles
    rotations = max([len(v) for v in roles.values()] + [len(v) for v in mids.values()]
                    + [len(v) for v in ends.values()] + ([len(library.coins[c])] if c else [1]))
    seen: set[str] = set()
    for r in range(rotations):
        for part, end in options:
            for position in positions:
                pick = {role: patches[r % len(patches)] for role, patches in roles.items()}
                sep = pick["sep"]
                coins = library.coins[c][r % len(library.coins[c])] if c else None
                plan = [library.lead]
                if position == "coins_after_lead":
                    plan += [coins, sep]
                for size in part:
                    plan += [mids[size][r % len(mids[size])], sep]
                if position == "coins_after_stomps":
                    plan += [coins, sep]
                if j:
                    for _ in range(j - 1):
                        plan += [pick["jump"], sep]
                    plan.append(pick["exit"])
                else:
                    plan.append(ends[end][r % len(ends[end])])
                digest = plan_sha256(plan)
                if digest in seen:
                    continue
                seen.add(digest)
                yield plan, {"rotation": r, "coin_position": position, "stomp_clusters": list(part),
                             "stomp_end": end, "plan_sha256": digest}


def run_stitch(
    library: PatchLibrary,
    target: Mapping[str, int],
    evaluate: Callable[[Level], EvalResult],
    remaining_calls: Callable[[], int],
    *,
    max_plans: int = DEFAULT_STITCH_PLANS,
    contract: str = "v1",
) -> tuple[FunctionalCandidate | None, EvalResult | None, dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    first_plan: list[Patch] | None = None
    plans = stitch_plans(library, target) if contract == "v1" else stitch_plans_v2(library, target)
    for plan, meta in plans:
        if len(attempts) >= max_plans or remaining_calls() <= 0:
            break
        first_plan = first_plan or plan
        level = assemble(plan, pad=PAD[contract])
        result = evaluate(level)
        ok = contract_exact(result, target, contract)
        audit = (result.info or {}).get("takeoff_audit")
        attempts.append({**meta, "width": level.width, "valid": bool(result.valid),
                         "observed": observed_tuple(result), "exact": ok,
                         **({"hidden_takeoffs": audit["hidden_takeoffs"]} if audit else {})})
        if ok:
            candidate = FunctionalCandidate(
                level, dict(target), "certified_patch_stitching",
                {"builder": "stitch", "contract": contract, "library_sha256": library.sha256,
                 "plan": plan_payload(plan), "plan_tokens": [p.token() for p in plan],
                 **meta, "width": level.width, "level_sha256": level_sha256(level)},
            )
            return candidate, result, {"stitch_attempts": attempts, "stitch_first_try": len(attempts) == 1,
                                       "first_plan": plan_payload(first_plan)}
    return None, None, {"stitch_attempts": attempts, "stitch_first_try": False,
                        "first_plan": plan_payload(first_plan) if first_plan else None,
                        "stitch_plans_exhausted": len(attempts) < max_plans}


# --------------------------------------------------------------------------
# (b) simulation-guided search over uncertified primitives


def role_families(contract: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in FAMILY_SETS[contract]:
        out.setdefault(FAMILIES[name].role, []).append(name)
    return out


def random_patch(rng: random.Random, role: str | None = None, contract: str = "v1") -> Patch:
    names = role_families(contract)[role] if role else sorted(FAMILY_SETS[contract])
    name = rng.choice(sorted(names))
    family = FAMILIES[name]
    for _ in range(256):
        params = {key: rng.randint(lo, hi) for key, (lo, hi) in sorted(family.ranges.items())}
        if family.valid_params(params):
            return Patch.make(name, **params)
    raise RuntimeError(f"could not sample valid params for {name}")


def tweak_patch(rng: random.Random, patch: Patch) -> Patch:
    family = FAMILIES[patch.family]
    params = patch.p
    for _ in range(64):
        key = rng.choice(sorted(family.ranges))
        lo, hi = family.ranges[key]
        trial = dict(params)
        trial[key] = min(hi, max(lo, params[key] + rng.choice((-2, -1, 1, 2))))
        if trial != params and family.valid_params(trial):
            return Patch.make(patch.family, **trial)
    return patch


def search_fitness(result: EvalResult, target: Mapping[str, int], contract: str = "v1") -> int:
    observed = observed_tuple(result)
    if not result.valid or float(result.metrics.get("complete", 0.0)) != 1.0 or observed is None:
        return INVALID_PENALTY
    error = sum(abs(observed[name] - int(target[name])) for name in DYNAMIC_NAMES)
    if contract == "v2":  # hidden takeoffs count as jump error
        error += int((result.info or {}).get("takeoff_audit", {}).get("hidden_takeoffs", 0))
    return error


def mutate(rng: random.Random, genome: list[Patch], observed: Mapping[str, int] | None,
           target: Mapping[str, int], contract: str = "v1") -> list[Patch]:
    child = list(genome)
    if observed is not None and rng.random() < 0.5:
        deltas = [(name, int(target[name]) - int(observed[name])) for name in DYNAMIC_NAMES
                  if int(target[name]) != int(observed[name])]
        if deltas:
            name, delta = rng.choice(deltas)
            if delta > 0 and len(child) < MAX_GENOME:
                child.insert(rng.randint(0, len(child)), random_patch(rng, name, contract))
                return child
            indices = [i for i, patch in enumerate(child) if patch.role == name]
            if delta < 0 and indices:
                del child[rng.choice(indices)]
                return child
    op = rng.choice(("insert", "delete", "replace", "tweak", "swap"))
    if op == "insert" or not child:
        if len(child) < MAX_GENOME:
            child.insert(rng.randint(0, len(child)), random_patch(rng, None, contract))
    elif op == "delete":
        del child[rng.randrange(len(child))]
    elif op == "replace":
        child[rng.randrange(len(child))] = random_patch(rng, None, contract)
    elif op == "tweak":
        index = rng.randrange(len(child))
        child[index] = tweak_patch(rng, child[index])
    elif len(child) >= 2:
        index = rng.randrange(len(child) - 1)
        child[index], child[index + 1] = child[index + 1], child[index]
    return child


def _width(genome: Sequence[Patch]) -> int:
    return sum(patch.width for patch in genome)


def target_seed(target: Mapping[str, int], seed: int) -> int:
    key = f"{seed}:" + ",".join(f"{name}={int(target[name])}" for name in DYNAMIC_NAMES)
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def run_search(
    target: Mapping[str, int],
    evaluate: Callable[[Level], EvalResult],
    remaining_calls: Callable[[], int],
    *,
    seed: int = SEED,
    initial: Sequence[Patch] = (),
    contract: str = "v1",
) -> tuple[FunctionalCandidate | None, EvalResult | None, dict[str, Any]]:
    rng = random.Random(target_seed(target, seed))
    pad = PAD[contract]
    init = list(initial)
    calls = 0
    restarts = 0
    trace: list[int] = []

    def score(genome: list[Patch]) -> tuple[int, EvalResult]:
        nonlocal calls
        result = evaluate(assemble(genome, pad=pad))
        calls += 1
        return search_fitness(result, target, contract), result

    def done(genome: list[Patch], result: EvalResult) -> tuple[FunctionalCandidate, EvalResult, dict[str, Any]]:
        level = assemble(genome, pad=pad)
        candidate = FunctionalCandidate(
            level, dict(target), "simulation_guided_patch_search",
            {"builder": "search", "contract": contract, "seed": seed, "seeded_from_stitch": bool(initial),
             "plan": plan_payload(genome), "plan_tokens": [p.token() for p in genome],
             "plan_sha256": plan_sha256(genome), "width": level.width,
             "level_sha256": level_sha256(level)},
        )
        return candidate, result, {"search_calls": calls, "search_restarts": restarts,
                                   "search_best_trace": trace}

    if remaining_calls() <= 0:
        return None, None, {"search_calls": 0, "budget_exhausted": True}
    current = list(init)
    current_fit, current_result = score(current)
    trace.append(current_fit)
    if current_fit == 0 and contract_exact(current_result, target, contract):
        return done(current, current_result)
    best_fit = current_fit
    stagnant = 0
    rejected = 0
    while remaining_calls() > 0 and rejected < 10_000:
        child = mutate(rng, current, observed_tuple(current_result), target, contract)
        if _width(child) > MAX_WIDTH or child == current:
            rejected += 1
            continue
        rejected = 0
        fit, result = score(child)
        if fit == 0 and contract_exact(result, target, contract):
            trace.append(fit)
            return done(child, result)
        if fit <= current_fit:
            current, current_fit, current_result = child, fit, result
        if current_fit < best_fit:
            best_fit, stagnant = current_fit, 0
            trace.append(best_fit)
        else:
            stagnant += 1
        if stagnant >= STAGNATION_RESTART and remaining_calls() > 0:
            restarts += 1
            stagnant = 0
            current = list(init)
            current_fit, current_result = score(current)
    return None, None, {"search_calls": calls, "search_restarts": restarts,
                        "search_best_trace": trace, "search_best_fitness": best_fit,
                        "budget_exhausted": True}


# --------------------------------------------------------------------------
# resolver adapter


class SmbDynamicBuilderAdapter:
    domain = "smb_dynamic"
    target_names = DYNAMIC_NAMES

    def __init__(self, mode: str = "stitch_then_search", *, library: Any = None,
                 stitch_plans: int = DEFAULT_STITCH_PLANS, seed: int = SEED,
                 pool: WidthEvaluatorPool | None = None, contract: str = "v1") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if contract not in CONTRACTS:
            raise ValueError(f"contract must be one of {CONTRACTS}")
        self.mode = mode
        self.contract = contract
        loader = PatchLibrary.load if contract == "v1" else PatchLibraryV2.load
        self.library = library if library is not None or mode == "search" else loader()
        self.stitch_plans = int(stitch_plans)
        self.seed = int(seed)
        self._pool = pool or WidthEvaluatorPool()
        self._envelope = get_smb_target_envelope("expanded")

    def validate(self, spec: FunctionalSpec) -> list[str]:
        errors: list[str] = []
        allowed = set(self.target_names)
        if set(spec.constraints) - allowed:
            errors.append(f"dynamic SMB received foreign metrics: {sorted(set(spec.constraints)-allowed)}")
        if not set(spec.constraints):
            errors.append("dynamic SMB requires at least one gameplay constraint")
        for name, constraint in spec.constraints.items():
            if name not in self._envelope:
                continue
            lower, upper = self._envelope[name]
            if not lower <= constraint.value <= upper:
                errors.append(f"{name}={constraint.value} outside expanded envelope [{lower},{upper}]")
        for objective in spec.objectives:
            if objective.metric not in allowed:
                errors.append(f"dynamic SMB objective uses foreign metric: {objective.metric}")
        return errors

    def lookup(self, spec: FunctionalSpec) -> FunctionalCandidate | None:
        return None

    def evaluate(self, level: Level) -> EvalResult:
        if self.contract == "v2":
            return self._pool.evaluate_audited(level)
        return self._pool.evaluate(level)

    def result_matches(self, result: EvalResult, selected_target: Mapping[str, int]) -> bool:
        return contract_exact(result, selected_target, self.contract)

    def select_target(self, spec: FunctionalSpec) -> dict[str, int] | None:
        exact = spec.exact_target(DYNAMIC_NAMES)
        if exact is not None:
            return exact
        return select_satisfying_target(spec, envelope_targets(), stable_metric_order=DYNAMIC_NAMES)

    def search(self, spec: FunctionalSpec, evaluate: Callable[[Level], EvalResult],
               remaining_calls: Callable[[], int]) -> SearchOutcome:
        target = self.select_target(spec)
        if target is None:
            return SearchOutcome(None, None, "no_target_in_envelope", {"mode": self.mode})
        details: dict[str, Any] = {"mode": self.mode, "contract": self.contract, "selected_target": target}
        initial: list[Patch] = []
        if self.mode in ("stitch", "stitch_then_search"):
            candidate, result, info = run_stitch(
                self.library, target, evaluate, remaining_calls, max_plans=self.stitch_plans,
                contract=self.contract)
            details.update(info)
            if candidate is not None:
                return SearchOutcome(candidate, result, candidate.strategy, details)
            if self.mode == "stitch":
                return SearchOutcome(None, None, "certified_patch_stitching", details)
            if info.get("first_plan"):
                initial = [Patch.from_dict(item) for item in info["first_plan"]]
        candidate, result, info = run_search(
            target, evaluate, remaining_calls, seed=self.seed, initial=initial, contract=self.contract)
        details.update(info)
        strategy = candidate.strategy if candidate is not None else "simulation_guided_patch_search"
        return SearchOutcome(candidate, result, strategy, details)


def build_resolver(mode: str = "stitch_then_search", **kwargs: Any) -> FunctionalResolver:
    from experiments.r108_smb_functional_harness import SmbStaticFunctionalAdapter

    resolver = FunctionalResolver()
    resolver.register(SmbDynamicBuilderAdapter(mode, **kwargs))
    resolver.register(SmbStaticFunctionalAdapter())
    return resolver
