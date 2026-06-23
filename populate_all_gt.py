# populate_all_gt.py
"""
Populate ground truth mechanisms for all domains.
"""

import json
import re
from pathlib import Path

def populate_scenarios(file_path):
    """Populate ground truth fields for all scenarios."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📂 Processing {len(data['scenarios'])} scenarios...")
    
    updated_count = 0
    for scenario in data['scenarios']:
        gt = scenario.get('causal_ground_truth', {})
        mechanism = gt.get('mechanism', '')
        category = scenario.get('category', 'Unknown')  # <-- FIX: Define category here
        
        # If mechanism is empty, extract from description or use domain template
        if not mechanism:
            description = scenario.get('description', '')
            
            # Try to extract causal chain from description
            mechanism = extract_mechanism_from_description(description, category)
            gt['mechanism'] = mechanism
        
        # Populate minimal_sufficient_set
        if mechanism and not scenario.get('minimal_sufficient_set'):
            steps = re.split(r' → | → |â†’ |â†’', mechanism)
            scenario['minimal_sufficient_set'] = [s.strip() for s in steps if s.strip()]
            updated_count += 1
        
        # Populate non_causal_correlates
        if not gt.get('non_causal_correlates'):
            desc = scenario.get('description', '').lower()
            non_causal = extract_non_causal(desc, category)
            gt['non_causal_correlates'] = non_causal
            updated_count += 1
        
        scenario['causal_ground_truth'] = gt
    
    # Save
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Updated {updated_count} fields in {len(data['scenarios'])} scenarios")

def extract_mechanism_from_description(description, category):
    """Extract causal mechanism from description."""
    # Domain-specific extraction patterns
    patterns = {
        'Finance': r'(?:triggered|due to|caused by|resulting from)\s+([^.]+?)(?:\.|,|;|$)',
        'Healthcare': r'(?:diagnosed|admitted|treated for|developed)\s+([^.]+?)(?:\.|,|;|$)',
        'Public Event': r'(?:due to|caused by|because of)\s+([^.]+?)(?:\.|,|;|$)',
        'Road Maintenance': r'(?:due to|caused by|after)\s+([^.]+?)(?:\.|,|;|$)',
        'Traffic Accident': r'(?:due to|caused by|after)\s+([^.]+?)(?:\.|,|;|$)',
        'Weather': r'(?:due to|caused by|because of)\s+([^.]+?)(?:\.|,|;|$)'
    }
    
    pattern = patterns.get(category, r'(?:due to|caused by)\s+([^.]+?)(?:\.|,|;|$)')
    match = re.search(pattern, description, re.IGNORECASE)
    
    if match:
        cause = match.group(1).strip()
        # Try to build a chain with more steps
        chain = f"{cause} → reaction → consequence → outcome"
        return chain
    
    # Default: use domain template with more specific wording
    templates = {
        'Finance': 'market event → investor reaction → price movement → consequence',
        'Healthcare': 'medical condition → intervention → patient response → outcome',
        'Public Event': 'crowd gathering → capacity exceeded → congestion → delay',
        'Road Maintenance': 'road hazard → driver reaction → loss of control → collision',
        'Traffic Accident': 'driver action → vehicle movement → collision → damage',
        'Weather': 'weather event → road condition change → vehicle response → incident'
    }
    
    return templates.get(category, 'cause → effect → consequence → outcome')

def extract_non_causal(description, category):
    """Extract non-causal correlates from description."""
    non_causal = []
    
    # Days of week
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    if any(day in description for day in days):
        non_causal.append('day_of_week')
    
    # Time periods
    times = ['morning', 'afternoon', 'evening', 'night', 'rush hour', 'peak hour']
    if any(time in description for time in times):
        non_causal.append('time_of_day')
    
    # Vehicle colors (if mentioned)
    colors = ['red', 'blue', 'white', 'black', 'silver', 'grey', 'gray', 'yellow', 'green']
    if any(color in description for color in colors):
        non_causal.append('vehicle_color')
    
    # Traffic context
    contexts = ['weekend', 'holiday', 'commuter', 'shopping', 'rush']
    if any(context in description for context in contexts):
        non_causal.append('traffic_context')
    
    # Weather conditions (if mentioned but not causal)
    weather_terms = ['sunny', 'clear', 'dry', 'warm', 'hot', 'pleasant']
    if any(term in description for term in weather_terms):
        non_causal.append('weather_context')
    
    return non_causal

if __name__ == "__main__":
    populate_scenarios('data/json/scenarios.json')