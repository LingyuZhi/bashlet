import json, subprocess, sys
from pathlib import Path
subprocess.run([sys.executable, 'tools/stock_value.py'], check=True)
assert json.loads(Path('reports/stock-value.json').read_text()) == {'oak': 375, 'elm': 0}
Path('data/stock.json').write_text('[{"sku": "ash", "qty": 2, "unit_cents": 77}]\n')
subprocess.run([sys.executable, 'tools/stock_value.py'], check=True)
assert json.loads(Path('reports/stock-value.json').read_text()) == {'ash': 154}
