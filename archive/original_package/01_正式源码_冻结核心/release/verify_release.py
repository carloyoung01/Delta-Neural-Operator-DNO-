"""Verify that this formal distribution reuses the frozen source verbatim.
Run in release root: python release/verify_release.py
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
REFERENCE='fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1'
actual=hashlib.sha256((ROOT/'dno.py').read_bytes()).hexdigest()
assert actual==REFERENCE, 'Core source differs from frozen release'
manifest=json.loads((ROOT/'release/FILE_MANIFEST.json').read_text())
for name, expected in manifest['sha256'].items():
    p=ROOT/name
    assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected, name
print(json.dumps({'core_matches_original':True,'files_verified':len(manifest['sha256']), 'core_sha256':actual},ensure_ascii=False,indent=2))
