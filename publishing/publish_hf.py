"""Publish an explicitly curated DNO research subset to a Hugging Face model repo.

Requires HF_TOKEN and HF_REPO_ID environment variables. Run from repository root.
Does not create repositories or delete remote files.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "publishing" / "huggingface" / "README.md"

APPROVED = [
    "dno.py",
    "demo.py",
    "train_synthetic.py",
    "verify_checkpoints.py",
    "requirements.txt",
    "requirements_research.txt",
    "LICENSE",
    "NOTICE",
    "docs/ARCHITECTURE.md",
    "docs/THEORY_TO_ARCHITECTURE.md",
    "docs/TRAINING_AND_DATA.md",
    "docs/EXPERIMENTS.md",
    "docs/VERIFICATION_REPORT.md",
    "docs/CODE_AND_EVIDENCE_MAP.md",
    "docs/PROVENANCE.md",
    "docs/RESEARCH_OUTLOOK.md",
    "evidence/pair_13.pt",
    "evidence/pair_29.pt",
    "evidence/history_17.pt",
    "evidence/history_31.pt",
    "evidence/training_results.json",
    "evidence/checkpoint_verification.json",
    "evidence/counterfactuals.json",
]


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not token or not repo_id:
        raise SystemExit("HF_TOKEN and HF_REPO_ID are required")
    if not CARD.is_file():
        raise SystemExit("Missing approved Hugging Face model card")
    missing = [name for name in APPROVED if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Required publication files missing: " + ", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="dno-hf-") as tmp:
        stage = Path(tmp)
        shutil.copy2(CARD, stage / "README.md")
        for name in APPROVED:
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        HfApi(token=token).upload_folder(
            folder_path=str(stage),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Sync curated DNO research files from GitHub",
        )
    print(f"Published curated DNO files to {repo_id}")


if __name__ == "__main__":
    main()
