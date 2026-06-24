# fix_ground_truth_for_c4.py
"""Populate non_causal_correlates for all scenarios."""

import json
import re

def fix_c4_ground_truth():
    """Add non_causal_correlates to all scenarios."""
    
    with open('data/json/scenarios.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for scenario in data['scenarios']:
        gt = scenario.get('causal_ground_truth', {})
        
        # Extract from description
        desc = scenario.get('description', '').lower()
        non_causal = []
        
        # Days of week
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        if any(d in desc for d in days):
            non_causal.append('day_of_week')
        
        # Time periods
        times = ['morning', 'afternoon', 'evening', 'night', 'rush hour']
        if any(t in desc for t in times):
            non_causal.append('time_of_day')
        
        # Colors
        colors = ['red', 'blue', 'white', 'black', 'silver', 'grey']
        if any(c in desc for c in colors):
            non_causal.append('vehicle_color')
        
        # Traffic context
        contexts = ['weekend', 'holiday', 'commuter', 'shopping']
        if any(c in desc for c in contexts):
            non_causal.append('traffic_context')
        
        gt['non_causal_correlates'] = non_causal
        scenario['causal_ground_truth'] = gt
    
    # Save
    with open('data/json/scenarios.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {len(data['scenarios'])} scenarios with non_causal_correlates")

if __name__ == "__main__":
    fix_c4_ground_truth()