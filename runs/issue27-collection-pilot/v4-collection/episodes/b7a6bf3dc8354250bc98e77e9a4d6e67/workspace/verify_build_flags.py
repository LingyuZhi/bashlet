import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, 'tools/build_flags.py'], check=True)
result = json.loads(Path('reports/build-flags.json').read_text())
assert isinstance(result, list)
assert result[0] == 'cache'
print('feedback passed')
