"""Domain-routed resolution of functional level constraints.

The resolver deliberately separates three concerns:

* a front end produces a :class:`FunctionalSpec` (an LLM is optional);
* a domain adapter owns construction and search strategies;
* the harness owns routing, bounded verifier execution, and accounting.

``evaluator_calls`` are compute/reproducibility measurements, not monetary
charges.  A bounded search miss is always ``unknown`` and never evidence that
the requested behavior is infeasible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Callable, Iterable, Mapping, Protocol

from toolusepcg.evaluator import EvalResult
from toolusepcg.level import Level


RELATIONS = frozenset({"exact", "min", "max"})
OBJECTIVE_DIRECTIONS = frozenset({"minimize", "maximize"})
RESOLUTION_STATUSES = frozenset(
    {"exact", "unknown", "unsupported", "invalid_request"}
)

EVALUATION_WITNESS_FIELDS = (
    "evaluation_mode",
    "simulation_called",
    "engine_executed",
    "engine_skip_reason",
    "engine_skip_reasons",
    "engine_gate_status",
    "actions",
    "locations",
    "game_events",
    "normalized_event_trace",
    "spawn_positions",
    "event_coordinate_semantics",
    "event_adapter_version",
    "engine_contract",
)


def _jsonable(value: Any) -> Any:
    """Convert evaluator witness values without retaining bulky engine arrays."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"unsupported evaluator witness value: {type(value).__name__}")


def _evaluation_witness(result: EvalResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        name: _jsonable(result.info[name])
        for name in EVALUATION_WITNESS_FIELDS
        if name in result.info
    }



@dataclass(frozen=True)
class MetricConstraint:
    relation: str
    value: int

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"unsupported constraint relation: {self.relation!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, int):
            raise TypeError("constraint value must be an integer")

    def matches(self, observed: int) -> bool:
        if self.relation == "exact":
            return observed == self.value
        if self.relation == "min":
            return observed >= self.value
        return observed <= self.value

    def as_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "value": self.value}


@dataclass(frozen=True)
class FunctionalObjective:
    metric: str
    direction: str

    def __post_init__(self) -> None:
        if self.direction not in OBJECTIVE_DIRECTIONS:
            raise ValueError(f"unsupported objective direction: {self.direction!r}")

    def as_dict(self) -> dict[str, str]:
        return {"metric": self.metric, "direction": self.direction}


@dataclass(frozen=True)
class FunctionalSpec:
    domain: str
    constraints: Mapping[str, MetricConstraint]
    objectives: tuple[FunctionalObjective, ...] = ()

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("functional spec requires a domain")
        if not self.constraints:
            raise ValueError("functional spec requires at least one constraint")
        if len({item.metric for item in self.objectives}) != len(self.objectives):
            raise ValueError("functional objectives must use unique metrics")

    def matches(self, target: Mapping[str, int]) -> bool:
        return all(
            name in target and constraint.matches(int(target[name]))
            for name, constraint in self.constraints.items()
        )

    def exact_target(self, names: Iterable[str]) -> dict[str, int] | None:
        ordered = tuple(names)
        if set(self.constraints) != set(ordered):
            return None
        if any(self.constraints[name].relation != "exact" for name in ordered):
            return None
        return {name: self.constraints[name].value for name in ordered}

    def as_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "constraints": {
                name: value.as_dict() for name, value in sorted(self.constraints.items())
            },
            "objectives": [item.as_dict() for item in self.objectives],
        }


def select_satisfying_target(
    spec: FunctionalSpec,
    targets: Iterable[Mapping[str, int]],
    *,
    stable_metric_order: Iterable[str],
) -> dict[str, int] | None:
    """Choose deterministically from targets satisfying constraints/objectives."""

    order = tuple(stable_metric_order)
    matches = [{name: int(row[name]) for name in order} for row in targets if spec.matches(row)]
    if not matches:
        return None

    def key(row: Mapping[str, int]) -> tuple[int, ...]:
        objective_key = tuple(
            int(row[item.metric])
            if item.direction == "minimize"
            else -int(row[item.metric])
            for item in spec.objectives
        )
        return objective_key + tuple(int(row[name]) for name in order)

    return min(matches, key=key)


@dataclass
class FunctionalCandidate:
    level: Level
    selected_target: dict[str, int]
    strategy: str
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchOutcome:
    candidate: FunctionalCandidate | None
    result: EvalResult | None
    strategy: str
    details: dict[str, Any] = field(default_factory=dict)


class FunctionalDomainAdapter(Protocol):
    domain: str
    target_names: tuple[str, ...]

    def validate(self, spec: FunctionalSpec) -> list[str]: ...

    def lookup(self, spec: FunctionalSpec) -> FunctionalCandidate | None: ...

    def evaluate(self, level: Level) -> EvalResult: ...

    def result_matches(
        self, result: EvalResult, selected_target: Mapping[str, int]
    ) -> bool: ...

    def search(
        self,
        spec: FunctionalSpec,
        evaluate: Callable[[Level], EvalResult],
        remaining_calls: Callable[[], int],
    ) -> SearchOutcome: ...


class EvaluatorBudgetExhausted(RuntimeError):
    pass


class _EvaluationLedger:
    def __init__(self, evaluator: Callable[[Level], EvalResult], limit: int) -> None:
        if limit < 0:
            raise ValueError("max_evaluator_calls cannot be negative")
        self._evaluator = evaluator
        self.limit = int(limit)
        self.calls = 0
        self.wall_time_ms = 0.0

    def remaining(self) -> int:
        return self.limit - self.calls

    def evaluate(self, level: Level) -> EvalResult:
        if self.calls >= self.limit:
            raise EvaluatorBudgetExhausted("functional resolver evaluator budget exhausted")
        started = time.perf_counter()
        try:
            return self._evaluator(level)
        finally:
            self.calls += 1
            self.wall_time_ms += (time.perf_counter() - started) * 1000.0


@dataclass
class FunctionalResolution:
    status: str
    domain: str
    source: str
    spec: FunctionalSpec
    selected_target: dict[str, int] | None
    level: Level | None
    observed_metrics: dict[str, Any] | None
    evaluation_info: dict[str, Any] | None
    strategy: str | None
    evaluator_calls: int
    evaluator_wall_time_ms: int
    search: dict[str, Any]
    provenance: dict[str, Any]
    errors: list[str]

    def __post_init__(self) -> None:
        if self.status not in RESOLUTION_STATUSES:
            raise ValueError(f"invalid resolution status: {self.status!r}")

    def as_dict(self, *, include_level: bool = True) -> dict[str, Any]:
        level_text = self.level.to_string() if self.level is not None else None
        payload = {
            "status": self.status,
            "domain": self.domain,
            "source": self.source,
            "spec": self.spec.as_dict(),
            "selected_target": self.selected_target,
            "observed_metrics": self.observed_metrics,
            "evaluation_info": self.evaluation_info,
            "strategy": self.strategy,
            "evaluator_calls": self.evaluator_calls,
            "evaluator_wall_time_ms": self.evaluator_wall_time_ms,
            "evaluator_accounting": "compute/reproducibility only; not monetary cost",
            "search": self.search,
            "provenance": self.provenance,
            "errors": self.errors,
            "level_sha256": (
                hashlib.sha256(level_text.encode()).hexdigest() if level_text is not None else None
            ),
        }
        if include_level:
            payload["level"] = level_text
        return payload


class FunctionalResolver:
    """Route a functional spec through one registered domain adapter."""

    def __init__(self) -> None:
        self._adapters: dict[str, FunctionalDomainAdapter] = {}

    def register(self, adapter: FunctionalDomainAdapter) -> None:
        if adapter.domain in self._adapters:
            raise ValueError(f"functional domain already registered: {adapter.domain}")
        self._adapters[adapter.domain] = adapter

    def domains(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def resolve(
        self,
        spec: FunctionalSpec,
        *,
        allow_search: bool = True,
        max_evaluator_calls: int = 64,
    ) -> FunctionalResolution:
        adapter = self._adapters.get(spec.domain)
        if adapter is None:
            return FunctionalResolution(
                status="unsupported", domain=spec.domain, source="none", spec=spec,
                selected_target=None, level=None, observed_metrics=None, strategy=None,
                evaluation_info=None,
                evaluator_calls=0, evaluator_wall_time_ms=0, search={}, provenance={},
                errors=[f"no functional adapter registered for domain {spec.domain!r}"],
            )
        errors = adapter.validate(spec)
        if errors:
            return FunctionalResolution(
                status="invalid_request", domain=spec.domain, source="none", spec=spec,
                selected_target=None, level=None, observed_metrics=None, strategy=None,
                evaluation_info=None,
                evaluator_calls=0, evaluator_wall_time_ms=0, search={}, provenance={},
                errors=errors,
            )

        ledger = _EvaluationLedger(adapter.evaluate, max_evaluator_calls)
        candidate = adapter.lookup(spec)
        lookup_drift: list[str] = []
        if candidate is not None:
            try:
                result = ledger.evaluate(candidate.level)
            except EvaluatorBudgetExhausted:
                result = None
            if result is not None and adapter.result_matches(result, candidate.selected_target):
                return self._resolution(
                    "exact", "archive_or_compiler_hit", spec, candidate, result, ledger,
                    search={"attempted": False}, errors=[],
                )
            lookup_drift.append("lookup candidate failed verifier")

        if not allow_search:
            return self._resolution(
                "unknown", "lookup_miss", spec, candidate, None, ledger,
                search={"attempted": False, "budget_exhausted": ledger.remaining() == 0},
                errors=lookup_drift,
            )

        try:
            outcome = adapter.search(spec, ledger.evaluate, ledger.remaining)
        except EvaluatorBudgetExhausted:
            outcome = SearchOutcome(
                candidate=None,
                result=None,
                strategy="budget_exhausted",
                details={"budget_exhausted": True},
            )
        exact = bool(
            outcome.candidate is not None
            and outcome.result is not None
            and adapter.result_matches(outcome.result, outcome.candidate.selected_target)
        )
        return self._resolution(
            "exact" if exact else "unknown",
            "bounded_search" if exact else "search_miss",
            spec,
            outcome.candidate,
            outcome.result,
            ledger,
            search={"attempted": True, "strategy": outcome.strategy, **outcome.details},
            errors=lookup_drift,
        )

    @staticmethod
    def _resolution(
        status: str,
        source: str,
        spec: FunctionalSpec,
        candidate: FunctionalCandidate | None,
        result: EvalResult | None,
        ledger: _EvaluationLedger,
        *,
        search: dict[str, Any],
        errors: list[str],
    ) -> FunctionalResolution:
        return FunctionalResolution(
            status=status,
            domain=spec.domain,
            source=source,
            spec=spec,
            selected_target=(candidate.selected_target if candidate else None),
            level=(candidate.level if candidate else None),
            observed_metrics=(dict(result.metrics) if result is not None else None),
            evaluation_info=_evaluation_witness(result),
            strategy=(candidate.strategy if candidate else search.get("strategy")),
            evaluator_calls=ledger.calls,
            evaluator_wall_time_ms=int(round(ledger.wall_time_ms)),
            search=search,
            provenance=(dict(candidate.provenance) if candidate else {}),
            errors=errors,
        )
