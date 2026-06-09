# checkers/c2_spatial.py
import re

class C2SpatialChecker:
    def __init__(self):
        self.name = "C₂ Spatial Relevance"
        self.max_distance_km = 2.0
    
    def check(self, scenario, explanation):
        """
        Check if causes are spatially relevant.
        
        Returns:
            dict with passed (bool), confidence (float), reason (str), details (dict)
        """
        mentioned_locations = self._extract_locations(explanation)
        scenario_locations = self._get_scenario_locations(scenario)
        
        if not scenario_locations:
            return {
                'checker': 'C2',
                'passed': True,
                'confidence': 0.5,  # Low confidence — can't verify
                'reason': 'No locations in scenario to verify against',
                'details': {'mentioned_locations': mentioned_locations}
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
                confidence *= 0.6  # Penalty per violation
            
            # Adjust confidence based on match quality
            confidence *= match_score
        
        # Adjust for extraction quality
        if len(mentioned_locations) == 0:
            confidence *= 0.5
        
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
                'confidence_score': confidence
            }
        }
    
    def _check_location_plausibility(self, mentioned_loc, scenario_locations, scenario):
        """Returns (is_plausible, reason, match_score)"""
        mentioned_text = mentioned_loc['text'].lower()
        
        best_match_score = 0.0
        best_match_name = None
        
        for sc_loc in scenario_locations:
            sc_name = sc_loc['name'].lower()
            
            # Exact match = 1.0
            if mentioned_text == sc_name:
                return True, f"Exact match: {sc_loc['name']}", 1.0
            
            # Partial match = score based on length ratio
            if mentioned_text in sc_name:
                score = len(mentioned_text) / len(sc_name)
                if score > best_match_score:
                    best_match_score = score
                    best_match_name = sc_name
            
            if sc_name in mentioned_text:
                score = len(sc_name) / len(mentioned_text)
                if score > best_match_score:
                    best_match_score = score
                    best_match_name = sc_name
        
        if best_match_score > 0.5:
            return True, f"Partial match: {mentioned_loc['text']} ↔ {best_match_name}", best_match_score
        
        # Check against description
        if 'description' in scenario and mentioned_text in scenario['description'].lower():
            return True, f"Mentioned in description", 0.7
        
        # Check common road terms
        common_terms = ['road', 'street', 'highway', 'lane', 'intersection']
        if any(term in mentioned_text for term in common_terms):
            return True, f"Common road term: {mentioned_loc['text']}", 0.6
        
        return False, f"Location '{mentioned_loc['text']}' not found", 0.0
    
    def _extract_locations(self, text):
        # ... existing extraction logic ...
        pass
    
    def _get_scenario_locations(self, scenario):
        # ... existing location logic ...
        pass