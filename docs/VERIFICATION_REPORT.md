# Distribution verification — 17 November 2026

A disposable copy of this distribution was used to run the specified tests.

```text
python -m pytest -q -p no:cacheprovider tests/test_final.py tests/test_modal_interfaces.py research_tests
162 passed in 2.37s
```

`python verify_checkpoints.py` passed for all four original saved model files on fresh samples:

| Checkpoint | Verification |
| --- | --- |
| `pair_13.pt` | Source 100%, target 100%, fresh MSE 0.0002888851 |
| `pair_29.pt` | Source 100%, target 100%, fresh MSE 0.0002791320 |
| `history_17.pt` | Delayed-bit accuracy 100%, no-first-input control 51.2695% |
| `history_31.pt` | Delayed-bit accuracy 100%, no-first-input control 51.8555% |

The distributed `dno.py` SHA-256 equals the original: `fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1`.

The four distributed `.pt` files were compared byte-for-byte to the originals in `archive/original_package/`.

The verification program writes `evidence/checkpoint_verification.json`; tests were intentionally run in a disposable copy to preserve the distribution's original file hashes.
