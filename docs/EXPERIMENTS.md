# Experimental results and verification

## 1. New verification of existing code

| Test group | Checks passed | What it checks |
|---|---:|---|
| `tests/test_final.py` | 32/32 | Current state, self-evolution, executed contact, structural effects and optional observation |
| `tests/test_modal_interfaces.py` | 16/16 | Numeric interfaces for image/audio/video/vector-shaped inputs |
| `research_tests/` | 114/114 | Optional research diagnostics, numerical observation and structural comparisons |
| **Combined** | **162/162** | Full specified automated test selection |

The current run output is in `verification/current_run/pytest.log`. The 114 optional research tests are **not** 114 trained DNO skills.

## 2. Newly reproduced training

Training was rerun using `train_synthetic.py` with seeds **13, 29** (pair-addressing) and **17, 31** (delayed one-bit information). The current `evidence/` directory contains newly generated files. The pair-addressing tests reached **100% source and target accuracy** in the two defined experiments; the one-bit task reached **100% accuracy** at each of the defined delays **3, 7 and 15**.

Details, exact MSE values and control accuracies: [Training and data](TRAINING_AND_DATA.md). All **four newly saved checkpoints passed** fresh load-and-evaluate verification.

## 3. Reproduced 64-seed structural counterfactual

| Measurement | Current result |
|---|---:|
| Random initializations | 64 |
| Positive-support carriers created by composition | 30/64 |
| Detectable state change after interaction, including after releasing a generated carrier | 64/64 |
| No automatic endpoint propagation during passive evolution, among created-carrier cases | 30/30 |
| Changed relation state affects subsequent addressing, among created-carrier cases | 30/30 |

These experiments inspect **untrained** structure behavior; a generated carrier does not automatically imply a meaningful learned relation.

## 4. What is demonstrated

The checks establish executable local evolution; interactions that rewrite participants and can form carriers; effective state consequences; addressing dependent on current structure; explicit observation and non-observation; and the supervised task outcomes above.

The numeric interface tests exercise input representations; full language-model training or image/audio/video semantic training is **not part of those particular tests**. Broader computational behavior is an open empirical direction, described in [Research outlook](RESEARCH_OUTLOOK.md).

## 5. Reproducibility

The test output, training log, checkpoint re-evaluation and counterfactual logs from the same current run are all under `verification/current_run/`. Structured numeric records are in `evidence/`.

Older experiments are preserved in `archive/original_package/` and are labeled historical. They have not been merged into the newly generated current metrics.
