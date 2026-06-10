# checkers/c2_spatial.py
import re

class C2SpatialChecker:
    def __init__(self):
        self.name = "C₂ Spatial Relevance"
        self.max_distance_km = 2.0
    
    def check(self, scenario, explanation):
        """
        Check if causes are spatially relevant.
        Uses structured_output.spatial_location if available.
        """
        # Extract structured data
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            spatial_location = structured.get('spatial_location', '')
            explanation_text = explanation.get('explanation', '')
        else:
            structured = {}
            spatial_location = ''
            explanation_text = explanation
        
        # Get scenario locations
        scenario_locations = self._get_scenario_locations(scenario)
        
        # Use structured location if available
        if spatial_location and scenario_locations:
            is_plausible, reason, score = self._check_location_match(
                spatial_location, scenario_locations
            )
            if not is_plausible:
                return {
                    'checker': 'C2',
                    'passed': False,
                    'confidence': score,
                    'reason': reason,
                    'details': {
                        'structured_location': spatial_location,
                        'used_structured': True,
                        'violations': [{'location': spatial_location, 'reason': reason}]
                    }
                }
            else:
                return {
                    'checker': 'C2',
                    'passed': True,
                    'confidence': score,
                    'reason': f"Location validated: {spatial_location}",
                    'details': {'used_structured': True, 'scenario_locations': scenario_locations}
                }
        
        # Fallback to free text parsing
        return self._check_free_text(explanation_text, scenario_locations, scenario)
    
    def _check_location_match(self, spatial_location, scenario_locations):
        """Check if structured location matches scenario"""
        loc_lower = spatial_location.lower()
        for sc_loc in scenario_locations:
            sc_name = sc_loc.get('name', '').lower()
            if loc_lower in sc_name or sc_name in loc_lower:
                score = min(1.0, len(loc_lower) / len(sc_name)) if sc_name else 0.8
                return True, f"Matches: {spatial_location}", score
        return False, f"Location '{spatial_location}' not found in scenario", 0.3
    
    def _check_free_text(self, text, scenario_locations, scenario):
        """Fallback to free text parsing (preserved from original)"""
        mentioned_locations = self._extract_locations(text)
        
        if not scenario_locations:
            return {
                'checker': 'C2',
                'passed': True,
                'confidence': 0.5,
                'reason': 'No locations in scenario to verify against',
                'details': {'mentioned_locations': mentioned_locations, 'used_structured': False}
            }
        
        violations = []
        confidence = 1.0
        
        for loc in mentioned_locations:
            is_plausible, reason, match_score = self._check_location_plausibility(
                loc, scenario_locations, scenario
            )
            
            if not is_plausible:
                violations.append({
                    'location': loc['text'],
                    'reason': reason
                })
                confidence *= 0.6
        
        passed = len(violations) == 0
        
        return {
            'checker': 'C2',
            'passed': passed,
            'confidence': round(confidence, 3),
            'reason': 'All locations spatially relevant' if passed else f'{len(violations)} spatial violation(s)',
            'details': {
                'mentioned_locations': mentioned_locations,
                'scenario_locations': [loc['name'] for loc in scenario_locations],
                'violations': violations,
                'used_structured': False
            }
        }
    
    def _extract_locations(self, text):
        """Extract location mentions using patterns"""
        patterns = [
            (r'on\s+([A-Za-z0-9\s]+(?:Road|Street|Highway|E\d+|SZR|Corniche))', 'road'),
            (r'at\s+([A-Za-z0-9\s]+(?:intersection|exit|roundabout|bridge))', 'intersection'),
            (r'near\s+([A-Za-z0-9\s]+(?:Mall|Mosque|Island|City|exit))', 'area'),
        ]
        
        locations = []
        for pattern, loc_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                locations.append({
                    'text': match.strip(),
                    'type': loc_type
                })
        return locations
    
    def _get_scenario_locations(self, scenario):
        """Extract all location info from scenario"""
        locations = []
        
        if 'context' in scenario and 'locations' in scenario['context']:
            locations = scenario['context']['locations']
        elif 'Location' in scenario:
            locations = [{'name': scenario['Location']}]
        elif 'location' in scenario:
            locations = [{'name': scenario['location']}]
        
        return locations
    
    def _check_location_plausibility(self, mentioned_loc, scenario_locations, scenario):
        """Check if mentioned location is plausible given scenario"""
        mentioned_text = mentioned_loc['text'].lower()
        
        for sc_loc in scenario_locations:
            sc_name = sc_loc['name'].lower()
            if mentioned_text == sc_name:
                return True, f"Exact match: {sc_loc['name']}", 1.0
            if mentioned_text in sc_name or sc_name in mentioned_text:
                return True, f"Partial match: {mentioned_loc['text']} ↔ {sc_loc['name']}", 0.7
        
        if 'description' in scenario and mentioned_text in scenario['description'].lower():
            return True, f"Mentioned in description", 0.6
        
        common_terms = ['road', 'street', 'highway', 'lane', 'intersection']
        if any(term in mentioned_text for term in common_terms):
            return True, f"Common road term: {mentioned_loc['text']}", 0.5
        
        return False, f"Location '{mentioned_loc['text']}' not found", 0.0