# Natural-language metric input

`main.py` accepts a shared metric-request format for Binary, BinaryDoor, Zelda,
Sokoban, Lode Runner and SMB. The selected config determines the game; language
input cannot silently switch environments. Metrics are implemented by the
existing evaluator, not generated as executable code by the model.

```bash
python main.py --config configs/zelda.yaml \
  --request '玩家到钥匙的最短路为20，钥匙到门为30，放3个敌人' \
  --request-dry-run

python main.py --config configs/smb.yaml \
  --request '通关过程中实际击杀2个敌人，收集3枚金币，从地面起跳1次'
```

`--request` uses the configured OpenAI-compatible endpoint, model and API key
(`--provider openai`). It makes one structured parser call before level generation.
`--request-dry-run` prints the parsed request and all resolved targets, without
constructing a level or running the optimizer. It still calls the parser for text.

Other parser providers can use the Python API or supply `--request-spec FILE`:

```json
{
  "domain": "zelda",
  "targets": [
    {"metric": "player_key", "mode": "exact", "value": 20},
    {"metric": "key_door", "mode": "exact", "value": 30}
  ],
  "issues": []
}
```

```bash
python main.py --config configs/zelda.yaml --request-spec request.json --request-dry-run
```

The same validation applies to model output and JSON input. Repeated fields,
unknown metrics, cross-domain input, unsupported operations, invalid numeric
values, and reported ambiguity stop execution. Schema-constrained generation
is followed by independent Python validation. This validates the structured
contract; it cannot guarantee that the model interpreted the user's prose correctly.

## Executable capabilities

| Domain | Exact targets | Also supports maximize |
|---|---|---|
| binary | path, num_connected_regions=1 | path |
| binarydoor | door_path, num_connected_regions=1 | door_path |
| zelda | player_key, key_door, solution_length, enemies | path metrics |
| sokoban | crates, targets | — |
| loderunner | gold, enemies, ladder, rope | ladder, rope |
| smb | enemies, coins, jumps | all three |

Zelda requires both component paths together, with the same mode, or the total
solution length. Switching path families removes the inherited alternative.
Sokoban updates crate and goal counts together and rejects conflicting counts.
SMB measures kills, collected coins and ground jumps during simulation; Zelda
and Lode Runner enemy counts are tile counts. Existing config validation enforces
Zelda component bounds and the expanded SMB envelope.

This adapter exposes only operations the current main-tree scoring path actually
implements. Range constraints (`min`/`max`), minimization, arbitrary new metrics,
and the separate SMB static resolver are not supported through this CLI.
The frozen SMB functional-resolver experiment remains separate and retains its
own range/objective support. Do not convert an inequality into an exact target.

Unspecified targets retain the selected config's defaults. `exact` replaces the
whole target entry, including any old maximize threshold. The optimizer's
existing scoring weights, playability checks, acceptance and termination rules
continue to apply; parsing a target does not guarantee generation success or
add automatic exact stopping to every domain. Inspect dry-run output before
running requests whose inherited defaults matter.

Requests cannot be combined with `--target` or `--resume-from`. A resulting run
can subsequently be resumed normally: resolved targets are in the saved config.
The parsed request and original text are also recorded in runtime arguments.
`--extra-instruction` remains prompt-only and does not create executable metrics.

## Extending and using the API

`toolusepcg.metric_input.DOMAINS` is the capability registry. A `DomainMetrics`
entry declares metric semantics, allowed modes and integer bounds, plus an
optional `prepare(targets, updates)` callback for domain-specific coupling.
Adding an input adapter requires existing evaluator/config/scorer support for
its metrics. The parser and schema builder need no domain-specific branches.

```python
from toolusepcg.metric_input import apply_metric_request, parse_metric_request

# Any injected client implementing chat.completions.create works here.
resolved_config, parsed = parse_metric_request(text, config, client)
# Or bypass the model for a pre-parsed request:
resolved_config = apply_metric_request(config, parsed)
```

Both APIs return a new config without mutating the original. Normal evaluator,
prompt, scoring and config persistence paths consume it directly.
