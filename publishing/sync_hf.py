"""Sync a curated set of DNO research files to the Hugging Face model repo."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "dno.py", "demo.py", "train_synthetic.py", "verify_checkpoints.py",
    "requirements.txt", "requirements_research.txt", "LICENSE", "NOTICE",
    "docs/ARCHITECTURE.md", "docs/THEORY_TO_ARCHITECTURE.md",
    "docs/TRAINING_AND_DATA.md", "docs/EXPERIMENTS.md",
    "docs/VERIFICATION_REPORT.md", "docs/CODE_AND_EVIDENCE_MAP.md",
    "docs/PROVENANCE.md", "docs/RESEARCH_OUTLOOK.md",
    "evidence/pair_13.pt", "evidence/pair_29.pt",
    "evidence/history_17.pt", "evidence/history_31.pt",
    "evidence/training_results.json",
    "evidence/checkpoint_verification.json", "evidence/counterfactuals.json",
]


def main():
    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not token or not repo_id:
        raise SystemExit("HF_TOKEN and HF_REPO_ID are required")
    card = ROOT / "publishing/huggingface/README.md"
    missing = [p for p in FILES if not (ROOT / p).is_file()]
    if not card.is_file() or missing:
        raise SystemExit(f"Missing publication inputs: {missing}")
    with TemporaryDirectory() as directory:
        stage = Path(directory)
        shutil.copy2(card, stage / "README.md")
        for name in FILES:
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        HfApi(token=token).upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(stage),
            commit_message="Sync curated DNO files from GitHub",
        )
    print("DNO Hugging Face sync completed")


if __name__ == "__main__":
    main()
