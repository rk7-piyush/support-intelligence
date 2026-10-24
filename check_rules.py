import json
from pathlib import Path

config_path = Path("config/intent_routing.json")

with config_path.open("r", encoding="utf-8") as file:
    config = json.load(file)

for index, rule in enumerate(config["rules"], start=1):
    print(f'{index}. {rule["department"]}: {rule["keywords"]}')