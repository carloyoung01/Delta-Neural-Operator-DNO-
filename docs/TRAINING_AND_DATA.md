# Training and generated data

## 1. Data source

The two supervised tasks use **synthetic samples generated at runtime** in `train_synthetic.py`. There is no external fixed language, video, image or audio corpus used for these experiments. Different random draws are made during optimization and evaluation.

## 2. Pair-addressing

`data_pair()` creates four carriers with permuted symbolic keys and random scalar values. A two-key query identifies the numerical difference to predict. Optimization minimizes MSE of the **final value difference**. The task's source/target identity indices are evaluated, but not used as direct loss labels.

`pair_forward()` uses the existing key and source/target query neural paths; this task does not train `compose()` to create new carriers.

Configuration: seeds **13, 29**; **450 steps**, Adam **0.008**, batch size **128**, held-out batch **2048**.

| Seed | MSE | Source accuracy | Target accuracy | MSE with query removed |
|---|---:|---:|---:|---:|
| 13 | 0.000274312 | 100% | 100% | 1.5158314 |
| 29 | 0.000300448 | 100% | 100% | 1.5783383 |

## 3. Delayed one-bit current-state task

`sequence_batch()` supplies one binary first mark and random later perturbations. The final label is predicted using shared DNO generator dynamics and a task-specific linear classifier. The training loss is final cross-entropy; no memory-anchor position label is supplied.

Configuration: seeds **17, 31**; **350 steps**, Adam **0.006**, batch **128**, training delays **2–7**, evaluated delays **3, 7, 15**, each evaluated on **1024** samples. A separate control removes the first mark.

| Seed | Delay 3 | Delay 7 | Delay 15 | No first mark |
|---|---:|---:|---:|---:|
| 17 | 100% | 100% | 100% | 50.3906% |
| 31 | 100% | 100% | 100% | 54.3945% |

## 4. Current checkpoint verification

The four current checkpoint files were **regenerated** with the above training code in this release workspace, and checked using fresh seeded data with `verify_checkpoints.py`.

| Checkpoint | Fresh MSE / accuracy | Additional control |
|---|---|---|
| `pair_13.pt` | MSE 0.000288885, source 100%, target 100% | Fresh pair examples |
| `pair_29.pt` | MSE 0.000279132, source 100%, target 100% | Fresh pair examples |
| `history_17.pt` | accuracy 100% | no-first-mark 51.2695% |
| `history_31.pt` | accuracy 100% | no-first-mark 51.8555% |

## 5. Reproduction and provenance

Run training without overwriting the packaged evidence:

```bash
python train_synthetic.py --pair-steps 450 --history-steps 350 --out /tmp/dno-training-run
```

For checkout-specific checkpoint validation, work on a scratch copy; `verify_checkpoints.py` writes `evidence/checkpoint_verification.json`.

The **previous** checkpoints and their reports remain under `archive/original_package/`; metrics in this document are drawn exclusively from the **new training** in `evidence/`. Device/runtime differences can cause small numerical changes. The generated checkpoint bytes are not the same as the historical ones; this is not hidden by their similar task results.
