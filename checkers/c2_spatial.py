# checkers/c2_spatial.py
import re
from difflib import SequenceMatcher

class C2SpatialChecker:
    def __init__(self):
        self.name = "C₂ Spatial Relevance"
        self.max_distance_km = 2.0
        
        # Synonym mapping for location names
        self.synonym_map = {
            'SZR': 'Sheikh Zayed Road',
            'E11': 'Abu Dhabi - Dubai Highway',
            'E95': 'Al Ain Road',
            'E311': 'Sheikh Mohammed bin Zayed Road',
            'Corniche': 'Corniche Road',
            'SZR': 'Sheikh Zayed Road',
        }
    
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
        
        # Get scenario locations with synonyms
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
        """Check if structured location matches scenario with fuzzy matching"""
        loc_lower = spatial_location.lower()
        best_match = None
        best_score = 0.0
        best_name = None
        
        for sc_loc in scenario_locations:
            sc_name = sc_loc.get('name', '').lower()
            
            # Exact match
            if loc_lower == sc_name:
                return True, f"Exact match: {spatial_location}", 1.0
            
            # Contains match
            if loc_lower in sc_name:
                score = len(loc_lower) / len(sc_name)
                if score > best_score:
                    best_score = score
                    best_name = sc_name
            
            if sc_name in loc_lower:
                score = len(sc_name) / len(loc_lower)
                if score > best_score:
                    best_score = score
                    best_name = sc_name
            
            # Fuzzy match for typos/variations
            ratio = SequenceMatcher(None, loc_lower, sc_name).ratio()
            if ratio > best_score and ratio > 0.7:
                best_score = ratio
                best_name = sc_name
        
        if best_score > 0.7:
            return True, f"Match ({best_score:.0%}): {spatial_location} ≈ {best_name}", best_score
        
        return False, f"Location '{spatial_location}' not found in scenario", 0.3
    
    def _check_free_text(self, text, scenario_locations, scenario):
        """Fallback to free text parsing"""
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
            else:
                confidence = confidence * 0.7 + match_score * 0.3
        
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
        """Extract location mentions using expanded patterns"""
        patterns = [
            (r'on\s+([A-Za-z0-9\s]+(?:Road|Street|Highway|E\d+|SZR|Corniche))', 'road'),
            (r'at\s+([A-Za-z0-9\s]+(?:intersection|exit|roundabout|bridge))', 'intersection'),
            (r'near\s+([A-Za-z0-9\s]+(?:Mall|Mosque|Island|City|exit))', 'area'),
            (r'along\s+([A-Za-z0-9\s]+(?:Road|Street|Highway))', 'road'),
            (r'between\s+([A-Za-z\s]+)\s+and\s+([A-Za-z\s]+)', 'segment'),
            (r'(?:northbound|southbound|eastbound|westbound)\s+on\s+([A-Za-z0-9\s]+(?:Road|Street|Highway))', 'directional'),
            (r'at\s+the\s+([A-Za-z\s]+)\s+(?:interchange|junction)', 'interchange'),
            (r'(?:near|by)\s+([A-Za-z\s]+(?:exit|entrance))', 'exit'),
        ]
        
        locations = []
        for pattern, loc_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    for m in match:
                        if m and len(m) > 3:
                            locations.append({
                                'text': m.strip(),
                                'type': loc_type
                            })
                else:
                    locations.append({
                        'text': match.strip(),
                        'type': loc_type
                    })
        
        # Remove duplicates
        seen = set()
        unique_locations = []
        for loc in locations:
            key = loc['text'].lower()
            if key not in seen:
                seen.add(key)
                unique_locations.append(loc)
        
        return unique_locations
    
    def _get_scenario_locations(self, scenario):
        """Extract locations with synonyms"""
        locations = []
        
        if 'context' in scenario and 'locations' in scenario['context']:
            locations = scenario['context']['locations'].copy()
        elif 'Location' in scenario:
            locations = [{'name': scenario['Location']}]
        elif 'location' in scenario:
            locations = [{'name': scenario['location']}]
        
        # Add synonyms
        expanded = []
        for loc in locations:
            expanded.append(loc)
            name = loc.get('name', '')
            for short, full in self.synonym_map.items():
                if short in name:
                    expanded.append({'name': full, 'type': 'synonym'})
                elif full in name:
                    expanded.append({'name': short, 'type': 'synonym'})
        
        return expanded
    
    def _check_location_plausibility(self, mentioned_loc, scenario_locations, scenario):
        """Check if mentioned location is plausible given scenario"""
        mentioned_text = mentioned_loc['text'].lower()
        best_score = 0.0
        best_match = None
        
        for sc_loc in scenario_locations:
            sc_name = sc_loc['name'].lower()
            
            # Exact match
            if mentioned_text == sc_name:
                return True, f"Exact match: {sc_loc['name']}", 1.0
            
            # Contains match
            if mentioned_text in sc_name:
                score = len(mentioned_text) / len(sc_name)
                if score > best_score:
                    best_score = score
                    best_match = sc_name
            
            if sc_name in mentioned_text:
                score = len(sc_name) / len(mentioned_text)
                if score > best_score:
                    best_score = score
                    best_match = sc_name
            
            # Fuzzy match
            ratio = SequenceMatcher(None, mentioned_text, sc_name).ratio()
            if ratio > best_score and ratio > 0.6:
                best_score = ratio
                best_match = sc_name
        
        if best_score > 0.7:
            return True, f"Match ({best_score:.0%}): {mentioned_loc['text']} ≈ {best_match}", best_score
        elif best_score > 0.5:
            return True, f"Partial match: {mentioned_loc['text']} ≈ {best_match}", best_score
        
        # Check against description
        if 'description' in scenario and mentioned_text in scenario['description'].lower():
            return True, f"Mentioned in description", 0.6
        
        # Check common road terms
        common_terms = ['road', 'street', 'highway', 'lane', 'intersection', 'bridge', 'underpass']
        if any(term in mentioned_text for term in common_terms):
            return True, f"Common road term: {mentioned_loc['text']}", 0.5
        
        return False, f"Location '{mentioned_loc['text']}' not found", 0.0