# clean_scenarios.py

import json
import re

def clean_scenarios(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all encoding issues
    fixes = [
        (r'â†’', '→'),
        (r'â€“', '–'),
        (r'â€™', "'"),
        (r'â€', '"'),
        (r'â€œ', '"'),
        (r'â€\u009d', '"'),
    ]
    
    for old, new in fixes:
        content = content.replace(old, new)
    
    # Parse and re-save
    data = json.loads(content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("✅ Fixed encoding in scenarios.json")

if __name__ == "__main__":
    clean_scenarios('data/json/scenarios.json')