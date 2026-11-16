"""Read-only integrity verification of formal DNO release files."""
from pathlib import Path
import json, hashlib

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'release' / 'SHA256_ALL_FILES.json'
KNOWN = 'fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1'
db = json.loads(MANIFEST.read_text(encoding='utf-8'))
assert db['frozen_core_sha256'] == KNOWN
assert hashlib.sha256((ROOT/'dno.py').read_bytes()).hexdigest() == KNOWN
assert hashlib.sha256((ROOT/'frozen/dno.py').read_bytes()).hexdigest() == KNOWN
for rel, digest in db['files_sha256'].items():
    path=ROOT/rel
    assert path.is_file(), f'missing: {rel}'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest, f'changed: {rel}'
actual={str(x.relative_to(ROOT)) for x in ROOT.rglob('*') if x.is_file() and
        not any(part in ('__pycache__', '.pytest_cache') for part in x.relative_to(ROOT).parts)
        and x != MANIFEST}
assert actual==set(db['files_sha256']), {'unexpected':sorted(actual-set(db['files_sha256'])),
                                       'missing':sorted(set(db['files_sha256'])-actual)}
print(json.dumps({'verified':len(db['files_sha256']), 'core_unchanged':True,
                  'sha256':KNOWN}, ensure_ascii=False))
