"""Validate the candidate package without modifying the frozen DNO kernel.
Run from any working directory: python 04_重测证据/verify_candidate.py
"""
from pathlib import Path
import json, hashlib, sys
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'04_重测证据'/'FILE_SHA256_MANIFEST.json').read_text('utf-8'))
issues=[]
for rel, expected in manifest['sha256'].items():
    p=root / rel
    if not p.is_file():
        issues.append(f'MISSING {rel}')
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
        issues.append(f'HASH MISMATCH {rel}')
core=root/'01_正式源码_冻结核心'/'dno.py'
if hashlib.sha256(core.read_bytes()).hexdigest()!='fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1':
    issues.append('FROZEN CORE SHA MISMATCH')
print(json.dumps({'verified': len(manifest['sha256'])-len(issues),
                  'total':len(manifest['sha256']),
                  'frozen_core_unchanged': 'FROZEN CORE SHA MISMATCH' not in issues,
                  'errors': issues},ensure_ascii=False,indent=2))
sys.exit(1 if issues else 0)
