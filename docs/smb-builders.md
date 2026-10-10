# Verified SMB patch construction

Fresh smoke and language CLI runs use PAPER-M8b ceiling-free patch stitching
(`stitch_v2`). The builder composes variable-width levels from certified local
patches and verifies the assembled level with A*. Kills and collected coins are
simulation events. Ground jumps exclude stomp bounces. Every physical ground
takeoff must match a JUMP event: hidden takeoffs reject an otherwise exact level.

Low ceilings in the earlier corridor patches could suppress JUMP events while
Mario physically took off. The old whole-level backbone resolver also covered
only a subset of the expanded envelope. Those implementations remain available
for historical comparisons; current runs use the audited v2 contract.

```bash
pcg-reproduce smoke --output runs/smoke
pcg-reproduce smb-dynamic --workers 4 --output runs/smb-v2
pcg-reproduce natural32 --dry-run
```

The CPU grid tests enemies 0–4 × collected coins 0–5 × ground jumps 0–6
(210 targets), then audits every successful trajectory and writes a summary.
The certified v2 stitching grammar resolves 204 targets. The six targets with
zero kills and zero jumps return `unknown`; this is a construction limitation,
not a proof of infeasibility. No model call is needed for the grid or smoke.

```bash
python -m experiments.smb_patch_library_v2 --workers 4 --output runs/patch_library_v2.json
python -m experiments.paper_m8_smb_dynamic_builders grid --arms stitch_v2 search_v2_b64 --workers 4 --output runs/grid.json
python -m experiments.paper_m1_smb_language structured-ceiling --manifest benchmarks/smb/template_test.json --builder stitch_v2 --output runs/ceiling.json
```

`stitch_then_search_v2` adds bounded simulator search after stitching misses.
`--builder legacy_r108` reproduces the old language execution path. Existing
Python APIs in `paper_m1_smb_language` retain their legacy default for historical
regression callers; pass `builder="stitch_v2"` for current construction.
New native LM Studio M6 campaigns also record and use `stitch_v2`. Historical
job files without a builder field retain their original execution path.
Primitive-action baseline and mechanic-operation suites retain their frozen
protocols. Static tile-count control uses its separate static adapter.

See [synchronization validation](sync-validation.md) for the local checks.

The bundled certification files include the fixed parameter grid, accepted and
rejected contexts, repeat checks and library hashes. They are development assets,
not held-out target levels. `docs/sync-provenance.json` records the source revision.
