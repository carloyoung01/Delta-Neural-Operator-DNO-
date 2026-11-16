# GitHub publication and versioning

**Repository:** https://github.com/carloyoung01/Delta-Neural-Operator-DNO-

**Published research release:** [v1.0.0](https://github.com/carloyoung01/Delta-Neural-Operator-DNO-/releases/tag/v1.0.0)

The `v1.0.0` tag identifies the original research release at commit `fe51245aa53e45cc8d4ec61ed248c9b265a2220e`. The `main` branch may receive subsequent documentation, integrity-manifest, and research-tool maintenance changes without changing that tag. Do not describe `main` as byte-identical to `v1.0.0` after maintenance.

## Inspect and run the source

- [Core source](dno.py) — executable DNO neural operator module.
- [Demonstration](demo.py) — small usage example.
- [Synthetic training](train_synthetic.py) — controlled training tasks.
- [Architecture](docs/ARCHITECTURE.md) — mathematical and computational description.

```bash
python -m pip install -r requirements.txt
python demo.py
python -m pip install -r requirements_research.txt
python -m pytest -q -p no:cacheprovider tests/test_final.py tests/test_modal_interfaces.py research_tests
python stress_audit.py --seeds 2
python verify_distribution.py
```

`stress_audit.py` is an optional observational stress utility. Its import uses the root `dno.py`, which has the same SHA-256 as the archived frozen core. This does not change the frozen DNO mathematics or trainable parameters.

The integrity manifest applies to the exact `main` tree after the corresponding maintenance commit. Avoid modifying tracked output files before running `verify_distribution.py`; verification scripts may regenerate files under `evidence/`. Run such scripts in a disposable checkout if preserving exact hashes matters.

To publish a future maintenance release, create a new tag on the tested commit and attach release notes. Do not move the existing `v1.0.0` tag.
