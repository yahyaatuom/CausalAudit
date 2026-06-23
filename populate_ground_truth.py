# populate_ground_truth.py
"""
Populate minimal_sufficient_set and non_causal_correlates
from existing mechanism strings.
"""

import json
import re
from pathlib import Path

def populate_scenarios(file_path):
    """Populate ground truth fields from mechanism strings."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for scenario in data['scenarios']:
        gt = scenario.get('causal_ground_truth', {})
        mechanism = gt.get('mechanism', '')
        
        # Populate minimal_sufficient_set from mechanism
        if mechanism and not scenario.get('minimal_sufficient_set'):
            # Split by arrows and clean
            steps = re.split(r' → | → |â†’ |â†’', mechanism)
            steps = [s.strip() for s in steps if s.strip()]
            scenario['minimal_sufficient_set'] = steps
        
        # Populate non_causal_correlates from description
        if not gt.get('non_causal_correlates'):
            desc = scenario.get('description', '').lower()
            non_causal = []
            
            # Check for temporal references
            temporal = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                       'saturday', 'sunday', 'morning', 'afternoon', 'evening', 'night']
            for word in temporal:
                if word in desc:
                    non_causal.append(word)
            
            # Check for traffic context
            context = ['rush hour', 'commuter', 'weekend', 'holiday']
            for word in context:
                if word in desc:
                    non_causal.append(word)
            
            # Check for vehicle colors
            colors = ['red', 'blue', 'white', 'black', 'silver', 'grey']
            for word in colors:
                if word in desc:
                    non_causal.append(word)
            
            gt['non_causal_correlates'] = non_causal
    
    # Save
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {len(data['scenarios'])} scenarios")
    return data

if __name__ == "__main__":
    populate_scenarios('data/json/scenarios.json')