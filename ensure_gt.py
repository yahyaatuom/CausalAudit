# ensure_ground_truth.py
"""Ensure all scenarios have populated ground truth."""

import json
import re
from pathlib import Path

def ensure_ground_truth():
    """Populate missing ground truth fields."""
    
    file_path = Path('data/json/scenarios.json')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    updated = 0
    for scenario in data['scenarios']:
        gt = scenario.get('causal_ground_truth', {})
        
        # Ensure mechanism exists
        if not gt.get('mechanism'):
            # Extract from description
            desc = scenario.get('description', '')
            category = scenario.get('category', 'Unknown')
            gt['mechanism'] = extract_mechanism(desc, category)
            updated += 1
        
        # Ensure minimal_sufficient_set exists
        if not scenario.get('minimal_sufficient_set') and gt.get('mechanism'):
            steps = re.split(r' → | → |â†’ |â†’', gt['mechanism'])
            scenario['minimal_sufficient_set'] = [s.strip() for s in steps if s.strip()]
            updated += 1
        
        # Ensure non_causal_correlates exists
        if not gt.get('non_causal_correlates'):
            desc = scenario.get('description', '').lower()
            gt['non_causal_correlates'] = extract_non_causal(desc)
            updated += 1
        
        scenario['causal_ground_truth'] = gt
    
    # Save
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {updated} fields in {len(data['scenarios'])} scenarios")

def extract_mechanism(desc, category):
    """Extract mechanism from description."""
    # Domain-specific templates
    templates = {
        'Traffic Accident': 'driver action → collision → outcome',
        'Weather': 'weather event → road condition → accident',
        'Road Maintenance': 'hazard → driver reaction → collision',
        'Public Event': 'event → congestion → delay',
        'Healthcare': 'condition → intervention → outcome',
        'Finance': 'market event → reaction → consequence'
    }
    
    # Try to find causal phrase
    patterns = [
        r'(?:due to|caused by|triggered by)\s+([^.]+)',
        r'(?:after|following)\s+([^.]+)',
        r'(?:because of)\s+([^.]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, desc, re.IGNORECASE)
        if match:
            cause = match.group(1).strip()
            return f"{cause} → chain reaction → outcome"
    
    return templates.get(category, 'cause → effect → outcome')

def extract_non_causal(desc):
    """Extract non-causal correlates from description."""
    non_causal = []
    
    # Days
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    if any(d in desc for d in days):
        non_causal.append('day_of_week')
    
    # Time
    times = ['morning', 'afternoon', 'evening', 'night', 'rush hour']
    if any(t in desc for t in times):
        non_causal.append('time_of_day')
    
    # Colors
    colors = ['red', 'blue', 'white', 'black', 'silver', 'grey']
    if any(c in desc for c in colors):
        non_causal.append('vehicle_color')
    
    return non_causal

if __name__ == "__main__":
    ensure_ground_truth()