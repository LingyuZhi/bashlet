import os, shutil, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    shutil.copytree('/workspace', directory + '/task')
    os.chdir(directory + '/task')
    for path in Path('.').rglob('*'):
        if path.is_file(): path.chmod(0o600)
    source = Path('_grading/cases.py').read_text()
    exec(compile(source, '_grading/cases.py', 'exec'))
