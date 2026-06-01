# generate_correct_json.py
import pandas as pd
import json
import os
from pathlib import Path

# Dynamic path to your Excel file (assuming it's in the same folder)
excel_path = Path(__file__).parent / "scenarios_perturbed.xlsx"
df = pd.read_excel(excel_path, sheet_name=0)

scenarios = []

for idx, row in df.iterrows():
    # Skip header row
    scenario_id = str(row.get('Scenario_ID', ''))
    if pd.isna(row.get('Scenario_ID')) or scenario_id == 'Scenario_ID' or scenario_id.startswith('Scenario_'):
        continue
    
    # Safely parse complexity level
    try:
        complexity_level = int(row.get('Complexity_Level', 2)) if not pd.isna(row.get('Complexity_Level')) else 2
    except (ValueError, TypeError):
        complexity_level = 2
    
    scenario = {
        "id": scenario_id,
        "category": str(row.get('Incident_Category', '')),
        "complexity_level": complexity_level,
        "description": str(row.get('Incident_Description', '')),
        "minimal_sufficient_set": [],
        "causal_ground_truth": {
            "primary_cause": str(row.get('Primary_Cause', '')),
            "mechanism": str(row.get('Mechanism_Description', ''))
        }
    }
    scenarios.append(scenario)
    print(f"Processing: {scenario_id}")

output = {"metadata": {"scenario_count": len(scenarios)}, "scenarios": scenarios}

# Ensure the directory exists
os.makedirs('data/json', exist_ok=True)

with open('data/json/scenarios.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved {len(scenarios)} scenarios to data/json/scenarios.json")