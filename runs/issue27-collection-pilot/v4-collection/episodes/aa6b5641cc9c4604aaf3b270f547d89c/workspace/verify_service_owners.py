import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, 'tools/service_owners.py'], check=True)
result = json.loads(Path('reports/service-owners.json').read_text())
assert result['api'] == 'sam'
print('feedback passed')
