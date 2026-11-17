---
license: apache-2.0
library_name: pytorch
tags:
- research
- neural-architecture
- experimental
- synthetic-data
---

# Delta Neural Operator (DNO)

**Author:** Carlo Young  
**License:** Apache-2.0  
**Source of truth:** https://github.com/carloyoung01/Delta-Neural-Operator-DNO-

DNO is an experimental trainable neural architecture investigating evolving effective states, autonomous computational carriers, and history-dependent interactions through shared learnable dynamics.

This Hugging Face repository mirrors a curated subset of the GitHub source, including core code, synthetic training and verification scripts, research documentation, and four synthetic-task checkpoints. It is **not a pretrained general-purpose language model**.

## Intended use
Architecture research, defined synthetic-task reproduction, and independent experimental extensions.

## Evidence and limitations
The documented GitHub research materials report passing automated tests and successful checkpoint evaluations in their specified synthetic settings. See `docs/EXPERIMENTS.md` and `docs/VERIFICATION_REPORT.md` for details, controls, and scope. These results do not establish general-purpose language capabilities, broad model superiority, or energy savings.

## Reproduction
Refer to the [complete GitHub repository](https://github.com/carloyoung01/Delta-Neural-Operator-DNO-) for the full source, test suite, dependencies, and provenance.

Checkpoint files use PyTorch serialization; load only artifacts from trusted sources and review loading practices before use.
