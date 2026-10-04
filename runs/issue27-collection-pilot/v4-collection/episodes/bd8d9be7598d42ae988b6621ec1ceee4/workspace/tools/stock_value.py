import json

with open('data/stock.json') as f:
    data = json.load(f)

results = {}
for item in data:
    sku = item['sku']
    qty = item['qty']
    unit_cents = item['unit_cents']
    results[sku] = qty * unit_cents

with open('reports/stock-value.json', 'w') as f:
    json.dump(results, f)
