"""Shared language front end for the metrics implemented by the PCG harness.

Domain adapters declare executable capabilities, not arbitrary evaluator code.
The returned Config uses the existing scoring, feedback and persistence paths.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping

from .config import Config, config_from_dict, config_to_dict


class MetricInputError(ValueError):
    """A request cannot safely be translated into executable targets."""


@dataclass(frozen=True)
class MetricDefinition:
    description: str
    modes: tuple[str, ...] = ("exact", "maximize")
    minimum: int = 0
    maximum: int | None = None


@dataclass(frozen=True)
class DomainMetrics:
    metrics: Mapping[str, MetricDefinition]
    rules: str = ""
    prepare: Callable[[dict, dict], None] | None = None


def _zelda(targets: dict, updates: dict) -> None:
    paths = {"player_key", "key_door", "solution_length"}
    if paths.intersection(updates):
        for name in paths:
            targets.pop(name, None)


def _sokoban(targets: dict, updates: dict) -> None:
    counts = [updates[name] for name in ("crates", "targets") if name in updates]
    if len(counts) == 2 and counts[0] != counts[1]:
        raise MetricInputError("Sokoban crates and targets must have equal counts")
    if counts:
        updates["crates"] = dict(counts[0])
        updates["targets"] = dict(counts[0])


DOMAINS: dict[str, DomainMetrics] = {
    "binary": DomainMetrics({
        "path": MetricDefinition("Longest shortest path between two empty tiles (maze diameter)"),
        "num_connected_regions": MetricDefinition("Connected passable regions", ("exact",), 1, 1),
    }),
    "binarydoor": DomainMetrics({
        "door_path": MetricDefinition("Shortest path between the two fixed doors"),
        "num_connected_regions": MetricDefinition("Connected passable regions", ("exact",), 1, 1),
    }),
    "zelda": DomainMetrics({
        "player_key": MetricDefinition("Shortest player-to-key path length", minimum=10, maximum=40),
        "key_door": MetricDefinition("Shortest key-to-door path length", minimum=10, maximum=40),
        "solution_length": MetricDefinition("Sum of player_key and key_door"),
        "enemies": MetricDefinition("Enemy tile count, not kills", ("exact",)),
    }, "Specify both path components together OR their total, never both families. "
       "Component modes must agree. One player, one key, one door and connectivity are fixed.", _zelda),
    "sokoban": DomainMetrics({
        "crates": MetricDefinition("Crate count; also sets equal goal tile count", ("exact",), 1),
        "targets": MetricDefinition("Goal tile count; also sets equal crate count", ("exact",), 1),
    }, "Crates and goal counts must agree. One player and solvability are fixed. "
       "Solution length is maximized by the existing scorer; exact length is unsupported.", _sokoban),
    "loderunner": DomainMetrics({
        "gold": MetricDefinition("Gold tile count, not collected gold", ("exact",), 1),
        "enemies": MetricDefinition("Enemy tile count (evaluator field: enemy)", ("exact",)),
        "ladder": MetricDefinition("Ladder tile count"),
        "rope": MetricDefinition("Rope tile count"),
    }, "One player and collection of all gold are fixed playability requirements."),
    "smb": DomainMetrics({
        "enemies": MetricDefinition("Actual kills during simulation, NOT enemy tiles", maximum=4),
        "coins": MetricDefinition("Actual coins collected during simulation, NOT coin tiles", maximum=5),
        "jumps": MetricDefinition("Ground-initiated jumps; excludes stomp bounces", maximum=6),
    }, "Completion and tube validity are fixed requirements. Static tile-count requests are unsupported."),
}


def request_schema(domain: str) -> dict[str, Any]:
    """Build a strict schema from the chosen domain's registered vocabulary."""
    if domain not in DOMAINS:
        raise MetricInputError(f"Unsupported domain: {domain}")
    return {
        "type": "object", "additionalProperties": False,
        "required": ["domain", "targets", "issues"],
        "properties": {
            "domain": {"type": "string", "enum": [domain, "unknown"]},
            "targets": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["metric", "mode", "value"],
                "properties": {
                    "metric": {"type": "string", "enum": list(DOMAINS[domain].metrics)},
                    "mode": {"type": "string", "enum": ["exact", "maximize"]},
                    "value": {"type": ["integer", "null"]},
                },
            }},
            "issues": {"type": "array", "items": {"type": "string"}},
        },
    }


def request_prompt(domain: str) -> str:
    schema = request_schema(domain)
    adapter = DOMAINS[domain]
    catalog = {
        name: {"meaning": metric.description, "modes": metric.modes,
               "minimum": metric.minimum, "maximum": metric.maximum}
        for name, metric in adapter.metrics.items()
    }
    return (
        f"Translate the user's level request into executable targets for {domain}. "
        "Return only JSON matching the schema. Understand any input language. "
        "Do not invent values or metrics. Omit unspecified targets; existing configured "
        "targets remain active. Use null value for maximize. Preserve explicit numbers "
        "even outside supported ranges; code validates them. List in issues every "
        "ambiguity, conflict, unsupported metric, unsupported operation (including "
        "minimum/maximum bounds or minimization), or subjective requirement that cannot "
        "be measured. Do not approximate inequalities with exact targets. Distinguish "
        "tile counts from gameplay events; ask for clarification via issues when unclear. "
        "Use domain=unknown for another game. Fixed validity rules need no target entry.\n"
        f"Domain rules: {adapter.rules}\nMetrics: {json.dumps(catalog)}\n"
        f"Schema: {json.dumps(schema)}"
    )


def apply_metric_request(config: Config, payload: Mapping[str, Any]) -> Config:
    """Validate a parsed request and return a new config; never mutate the input."""
    domain = config.env.problem_type
    if domain not in DOMAINS:
        raise MetricInputError(f"Unsupported domain: {domain}")
    if not isinstance(payload, dict) or set(payload) != {"domain", "targets", "issues"}:
        raise MetricInputError("Expected exactly domain, targets and issues")
    if payload["domain"] != domain:
        raise MetricInputError(f"Request domain must match configured domain {domain}")
    issues = payload["issues"]
    if not isinstance(issues, list) or any(not isinstance(x, str) for x in issues):
        raise MetricInputError("issues must be a list of strings")
    if issues:
        raise MetricInputError("Request needs clarification or is unsupported: " + "; ".join(issues))
    entries = payload["targets"]
    if not isinstance(entries, list) or not entries:
        raise MetricInputError("Request must contain at least one executable target")
    adapter = DOMAINS[domain]
    updates = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"metric", "mode", "value"}:
            raise MetricInputError("Each target requires exactly metric, mode and value")
        name, mode, value = entry["metric"], entry["mode"], entry["value"]
        if not isinstance(name, str) or name not in adapter.metrics:
            raise MetricInputError(f"Unsupported metric for {domain}: {name!r}")
        if name in updates:
            raise MetricInputError(f"Duplicate target: {name}")
        metric = adapter.metrics[name]
        if mode not in metric.modes:
            raise MetricInputError(f"Unsupported mode for {name}: {mode!r}")
        if mode == "exact":
            if type(value) is not int:
                raise MetricInputError(f"{name} requires an integer value")
            if value < metric.minimum or (metric.maximum is not None and value > metric.maximum):
                raise MetricInputError(f"{name}={value} outside [{metric.minimum}, {metric.maximum}]")
        elif value is not None:
            raise MetricInputError(f"Maximize {name} requires value=null")
        updates[name] = {"mode": "target" if mode == "exact" else mode, "value": value}
    data = config_to_dict(config)
    targets = data["targets"]
    if adapter.prepare:
        adapter.prepare(targets, updates)
    # Replace whole target entries: do not inherit maximize thresholds for exact goals.
    targets.update(updates)
    try:
        return config_from_dict(data)
    except (TypeError, ValueError) as exc:
        raise MetricInputError(str(exc)) from exc


def parse_metric_request(text: str, config: Config, client: Any) -> tuple[Config, dict]:
    """Parse through an injected OpenAI-compatible client, then validate locally."""
    if not text.strip():
        raise MetricInputError("Empty metric request")
    domain = config.env.problem_type
    response = client.chat.completions.create(
        model=config.llm.model,
        messages=[{"role": "system", "content": request_prompt(domain)},
                  {"role": "user", "content": text}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "pcg_metric_request", "strict": True, "schema": request_schema(domain),
        }},
        temperature=0, max_tokens=config.llm.max_tokens,
        **({"extra_body": config.llm.extra_body} if config.llm.extra_body else {}),
    )
    choice = response.choices[0]
    if choice.finish_reason != "stop" or getattr(choice.message, "refusal", None):
        raise MetricInputError("Metric parser did not return a complete answer")
    try:
        payload = json.loads(choice.message.content)
    except (TypeError, ValueError) as exc:
        raise MetricInputError("Metric parser returned invalid JSON") from exc
    return apply_metric_request(config, payload), payload
