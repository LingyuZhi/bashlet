import json
from pathlib import Path
from collections import Counter
rows = json.loads(Path('data/stock.json').read_text())
result = {r['sku']: r['qty'] * r['unit_cents'] for r in rows}
Path('reports').mkdir(exist_ok=True)
Path('reports/stock-value.json').write_text(json.dumps(result, sort_keys=True) + '\n')
