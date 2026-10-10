# Core synchronization validation

Source: `PromptEvoAgenticPCG` commit `366c27822f13a35046a7041249a3f61671330aad`.
Validation performed on October 9, 2026 (America/New_York), with Python 3.12.

- Complete release regression suite: **228 passed, 3 skipped**, plus 252 passing subtests.
  Skips require optional Transformers serving qualification or xgrammar.
- Offline smoke: Zelda paths 17/31; SMB 3 kills, 5 collected coins, 1 ground jump;
  v2 physical takeoff audit clean.
- Expanded SMB envelope: **204/210 exact and deterministic**; all 204 successful
  trajectories have zero hidden takeoffs. The six zero-kill/zero-jump targets
  remain `unknown`. Development subset: 29/29.
- Structured language execution with `stitch_v2`: validation 29/29, test 38/38,
  Natural-32 valid requests 24/24; every successful trajectory passes the audit.
  These use ground-truth structured fields and make no new model parser calls.
- Native LM Studio campaign CPU check: first M6 request resolves through the
  recorded v2 builder; no process failures.
- Wheel and source distribution build successfully. The wheel includes certified
  patch assets and new modules. Extracted wheel runs audited SMB construction and
  metric-request validation from outside both repository checkouts.
- Metric-request CLI JSON dry-run and all named suite CLI argument plans pass.
- The final release includes the previously pending asset-path, packaging and
  experiment-test corrections. Asset paths resolve from the bundled benchmark
  package without a user name or local checkout path. No model inference or
  checkpoint download was performed during validation.

Regenerate the main offline checks from the release checkout:

```bash
python -m pytest -q
pcg-reproduce smoke --output runs/smoke
pcg-reproduce smb-dynamic --workers 4 --output runs/smb-v2
python -m experiments.paper_m1_smb_language structured-ceiling --manifest benchmarks/smb/template_validation.json --builder stitch_v2 --output runs/validation-v2.json
```
