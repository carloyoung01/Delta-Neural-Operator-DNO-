# Sources, consistency and provenance

## One executable release

The top-level `dno.py` is the authoritative runnable core. Source SHA-256:

`fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1`

`train_synthetic.py`, `verify_checkpoints.py`, `probe_counterfactuals.py`, core tests and examples are the matching executable programs. The original core, training program, test programs, examples and diagnostic scripts were **not rewritten** as part of this documentation and data-alignment rebuild.

## Current data versus past data

- `evidence/`: **new reproduction**. It contains four newly trained PyTorch checkpoints, newly generated `training_results.json`, `checkpoint_verification.json`, and `counterfactuals.json`.
- `verification/current_run/`: outputs of the same new test/train/verification run. See `RUN_METADATA.json` for commands and software versions.
- `archive/original_package/`: intact historical source package, original weights, results, math/Chinese research records, and all 21 legacy research archives within its `02_研究历史归档/` subdirectory.
- `verification/previous_audit/`: separate prior audit records; not inputs to the current performance tables.

The previous weights and current weights have **different SHA-256 hashes**. Both trace to the same training source and configured seeds/steps, with small numerical differences; do not claim the files are byte-identical or merge their measurement tables.

## Mathematical interpretation

The set of potential successors is a mathematical description of current-state-conditional operations, not a separate stored unknown-state collection or an enumerated search schedule. External multiview diagnostics are optional analysis files, not mandatory DNO compute phases. Observation and its absence are directly supported by the current runtime operations.

## Package validation

`MANIFEST_SHA256.json` covers all distribution files except itself. `verify_distribution.py` checks hashes and refuses missing/unexpected files. Training or checkpoint verification may write to `evidence/`; do those actions on a working copy, then regenerate a new manifest for any intentionally changed distribution.
