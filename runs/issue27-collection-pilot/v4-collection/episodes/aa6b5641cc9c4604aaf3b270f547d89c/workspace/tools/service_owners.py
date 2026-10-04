import json

def main():
    with open('config/services.json', 'r') as f:
        services = json.load(f)
    
    with open('config/teams.json', 'r') as f:
        teams = json.load(f)
    
    result = {}
    for service, team_code in services.items():
        if team_code in teams:
            result[service] = teams[team_code]
        else:
            result[service] = None
    
    with open('reports/service-owners.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

if __name__ == '__main__':
    main()
