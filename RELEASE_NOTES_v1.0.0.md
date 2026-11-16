# DNO v1.0.0 — Research release

**Author:** Carlo Young  
**License:** Apache License 2.0

Delta Neural Operator is a neural computation architecture in which effective current state carries the consequences of prior interactions. Relatively autonomous carriers evolve under shared learnable generation; actual interactions can modify them and create new carriers that participate in subsequent computation. Observation can occur, change, or remain absent.

## Included and verified

- A consistent executable core and supporting training/test scripts.
- A **newly regenerated** four-checkpoint synthetic training run, with fresh evaluation logs and independent checkpoint reloading.
- **162/162 automated checks passed**, comprising 32 core, 16 numeric interface and 114 independent research checks.
- **64-seed counterfactual reproduction:** 30/64 cases formed a positive-support carrier, and 64/64 retained a detectable effect of interaction on current-state evolution.
- Code-to-mathematics mapping, training/data description, current evidence records, source provenance, and archived earlier experiments.

## Research directions

The architecture can be studied at larger compute budgets, on more varied datasets, across different neural model families and in language, video, image, audio and other structured tasks. Additional effects are to be established through training and observation, rather than assumed in advance.

## Data distinction

`evidence/` holds **newly trained weights and current results**. `archive/original_package/` holds earlier model weights and documents, kept as historical sources. Old and new checkpoint results are not mixed. The full GitHub repository contains the current executable source and a separately archived original package.

**Core SHA-256:** `fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1`.

See `README.md`, `docs/ARCHITECTURE.md`, `docs/TRAINING_AND_DATA.md`, `docs/EXPERIMENTS.md`, `docs/PROVENANCE.md` and `docs/CODE_AND_EVIDENCE_MAP.md`.
