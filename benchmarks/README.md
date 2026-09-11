# Fixed inputs and runtime assets

These files are sufficient to recreate the paper tasks without copying the original run directories. Keep them immutable when reporting the registered protocol. Newly generated levels, traces and scores belong in a run directory.

| Input | Purpose |
| --- | --- |
| `zelda/targets.json` | Original 961-target geometry domain and fixed 120/120/721 development/validation/test partition |
| `zelda/test721.json` | Frozen test-case rows, including target IDs, seeds and initial-map hashes; Z64 is selected using the fixed ID list in `configs/` |
| `smb/targets.json` | 29 development controls used by the primitive-action and mechanic experiments |
| `smb/splits.json` | Frozen 96-case dynamic contract partition: 29 development, 29 validation, 38 test |
| `smb/language.json` | Original exposed language-contract examples and guards needed by preflights and regression checks |
| `smb/template_validation.json`, `smb/template_test.json` | Frozen bilingual language inputs, expected typed fields and guard dispositions |
| `smb/static_language.json` | Seven static-control requests and one guard |
| `smb/backbones/` | Exactly 16 development levels used as runtime mechanic templates |
| `smb/backbone_sources.json` | Only the provenance rows for those 16 templates, used by library preflight |
| `smb/search_seeds/` | Compact development seed maps and their index for bounded simulator search |
| `smb/engine_probe.json` | Minimal engine checks and seeds extracted from the base framework specification |

Natural-32, BinaryDoor-32, M7 tasks and scene descriptions live in `configs/`. Their expected fields are benchmark ground truth, not model outputs. The small backbone and search-seed maps are execution dependencies; removing them would change the resolver. Full atlases, agent result records, inference traces and historical aggregate scores are omitted.

## Regenerating inputs

To regenerate Zelda inputs in a separate directory:

```bash
python scripts/build_zelda_validation.py --output runs/materialized/validation120.json
python scripts/build_zelda_test.py --output runs/materialized/test721.json
```

The materializers check the original target-manifest identity and regenerate initial-map hashes. They do not draw a new target split. To inspect the deterministic template generator without replacing the distributed files, use `python -m experiments.paper_m1_smb_language build-manifests --output runs/materialized-language`.
