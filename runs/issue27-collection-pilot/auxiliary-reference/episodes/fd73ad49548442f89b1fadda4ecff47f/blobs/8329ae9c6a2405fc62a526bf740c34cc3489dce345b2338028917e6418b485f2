import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, 'tools/stock_value.py'], check=True)
result = json.loads(Path('reports/stock-value.json').read_text())
assert result['oak'] == 375
print('feedback passed')
