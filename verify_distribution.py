"""Verify complete DNO GitHub distribution without modifying it."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
expected=json.loads((ROOT/'MANIFEST_SHA256.json').read_text('utf-8'))
errors=[]
for rel,digest in expected['sha256'].items():
    path=ROOT/rel
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        errors.append(rel)
actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name != 'MANIFEST_SHA256.json' and '.git' not in p.parts and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
extras=sorted(actual-set(expected['sha256']))
print(json.dumps({'verified_files':len(expected['sha256'])-len(errors),'expected_files':len(expected['sha256']),'errors':errors,'extra_files':extras},indent=2))
raise SystemExit(1 if errors or extras else 0)
