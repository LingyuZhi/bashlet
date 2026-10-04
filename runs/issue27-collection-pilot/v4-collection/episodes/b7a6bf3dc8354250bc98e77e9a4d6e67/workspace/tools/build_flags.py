import json
from pathlib import Path

def main():
    with open('config/build.json', 'r') as f:
        config = json.load(f)
    
    result = []
    for key, value in config.items():
        if value is True:
            result.append(key)
    
    result.sort()
    
    with open('reports/build-flags.json', 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()
