"""Sync a curated set of DNO research files to the Hugging Face model repo."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

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
    api = HfApi(token=token)
    identity = api.whoami(token=token)
    account = identity.get("name", "unknown")
    print(f"Hugging Face token identity: {account}", flush=True)
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", token=token)
        print(f"Target model repository: {info.id}", flush=True)
    except HfHubHTTPError as error:
        raise SystemExit(
            f"Cannot access target model repository {repo_id} as {account}. "
            "Check the repository name and fine-grained token repository permissions. "
            f"HTTP status: {error.response.status_code}"
        ) from None
    try:
        _publish(api, repo_id, card)
    except (HfHubHTTPError, ConnectionError) as error:
        raise SystemExit(
            f"Hugging Face rejected upload to {repo_id} for token identity {account}. "
            "HTTP 403 at xet-write-token usually indicates insufficient write permission. "
            "In Hugging Face token settings, enable 'Write contents/settings of your repos' "
            "or 'Write contents/settings of selected repos' for this exact model repository. "
            "After changing token permissions, update the GitHub HF_TOKEN secret if a new "
            "token was created. Do not disclose the token. "
            f"Original error type: {type(error).__name__}"
        ) from None


def _publish(api, repo_id, card):
    with TemporaryDirectory() as directory:
        stage = Path(directory)
        shutil.copy2(card, stage / "README.md")
        for name in FILES:
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(stage),
            commit_message="Sync curated DNO files from GitHub",
        )
    print("DNO Hugging Face sync completed")


if __name__ == "__main__":
    main()
