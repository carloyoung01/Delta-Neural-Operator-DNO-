"""Run the original checkpoint test in an isolated temporary source copy.
Preserves all checksum-protected files of the candidate distribution.
"""
from pathlib import Path
import tempfile, subprocess, sys, shutil, hashlib
root=Path(__file__).resolve().parents[1]
src=root/'01_正式源码_冻结核心'
with tempfile.TemporaryDirectory(prefix='dno_readonly_') as td:
    cp=Path(td)/'src'
    shutil.copytree(src,cp,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache'))
    result=subprocess.run([sys.executable,'verify_checkpoints.py'],cwd=cp,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    digest=hashlib.sha256((cp/'dno.py').read_bytes()).hexdigest()
    assert digest=='fea8d9ea81c94dbf926b984b3c158f301afc71782f989c73b3c9bd02fac767b1'
print('PASS: 4/4 saved checkpoints verified from an isolated copy; candidate untouched.')
