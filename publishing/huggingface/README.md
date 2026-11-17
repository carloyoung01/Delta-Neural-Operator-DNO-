---
license: apache-2.0
library_name: pytorch
tags:
- research
- neural-architecture
- pytorch
- synthetic-data
- experimental
model-index:
- name: Delta Neural Operator (DNO) synthetic research checkpoints
  results: []
---

# Delta Neural Operator (DNO)

**Author:** Carlo Young  
**License:** Apache-2.0  
**Source:** https://github.com/carloyoung01/Delta-Neural-Operator-DNO-

DNO is an experimental, trainable neural computation architecture based on effective current state, shared learnable dynamics, and history-dependent interactions among computational carriers. This repository mirrors a **curated research subset** of the GitHub source at a selected revision.

## Intended use

Research, reproduction, and extension of the architecture and its defined synthetic experiments. The included checkpoints are task-specific experimental artifacts, **not** pretrained general-purpose language, image, audio, or video generation models.

## Architecture and evidence

The source repository documents carrier state, local evolution, conditional addressing, composition, observation, and current-state compaction. The broader Delta theoretical framework is research context; not every theoretical construct is a separate runtime operation.

The **GitHub v1.0.0 release** reports 162/162 automated tests passed, 4/4 synthetic checkpoints verified, 100% source/target selection on two defined pair-addressing evaluations, and 100% accuracy at evaluated delays 3, 7, and 15 for two delayed-bit runs. These are **reported results under specified experimental conditions**, not general performance guarantees. See the linked GitHub verification documents for the evaluation conditions and controls.

## Contents

- `dno.py`: core executable research architecture
- `train_synthetic.py`, `verify_checkpoints.py`: synthetic training and verification
- `demo.py`: example entry point
- `docs/`: selected mathematical, training, verification, and provenance documentation
- `evidence/`: four PyTorch research checkpoints and associated JSON records

**Security note:** The `.pt` checkpoint files use PyTorch serialization. Only load artifacts from sources you trust; review the verification code and use safe loading options where supported.

## Reproduction

See the [GitHub repository](https://github.com/carloyoung01/Delta-Neural-Operator-DNO-) for full source, tests, requirements, historical evidence, and exact reproduction instructions. The Hugging Face mirror is not a substitute for the full GitHub distribution.

## Limitations and future research

Current published evaluations are synthetic and bounded. No broad multimodal training, large-scale language benchmark superiority, or energy-efficiency advantage has been established by these experiments. Scaling, multimodal training, and compute/energy comparisons remain research directions.
