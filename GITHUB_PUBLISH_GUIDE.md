# Publishing to GitHub

This directory is a complete, self-contained distribution intended to be committed to the existing `carloyoung01/Delta-Neural-Operator` repository, without replacing its history.

1. Download and extract the complete ZIP locally.
2. Copy the extracted **contents** into a working clone of the repository, reviewing any pre-existing files before committing. In particular, do not force-push or erase commit history.
3. Run `python verify_distribution.py` and `python -m pytest -q -p no:cacheprovider tests/test_final.py tests/test_modal_interfaces.py research_tests` before publishing.
4. Commit files including `evidence/` and `archive/original_package/` (the archive includes older ZIPs). Check GitHub size limits for the archive files; use Git LFS if any file exceeds GitHub's accepted single-file size (current files are small).
5. Tag `v1.0.0` at the intended commit, create a GitHub Release and paste `RELEASE_NOTES_v1.0.0.md` as the release body. Attach the full package ZIP as a release asset if desired.

The files were prepared locally; preparing this ZIP does **not** itself create a GitHub commit, tag or release.
