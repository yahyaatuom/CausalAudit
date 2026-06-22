# checkers/c4_spurious.py
import re
import json
from pathlib import Path

class C4SpuriousChecker:
    def __init__(self, kb_path='data/spurious_patterns.json'):
        self.name = "C₄ Non-Spuriousness"
        
        kb_path = Path(__file__).parent.parent / kb_path
        try:
            with open(kb_path, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)['patterns']
        except FileNotFoundError:
            # Default patterns if file doesn't exist
            self.patterns = [
                {
                    "name": "temporal_correlate",
                    "regex": r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|evening|night|rush hour)",
                    "reason": "Temporal correlates are often spurious unless causally linked"
                },
                {
                    "name": "color_correlate",
                    "regex": r"(red|blue|white|black|silver|grey|yellow|green)",
                    "reason": "Vehicle color is rarely causal in accidents"
                },
                {
                    "name": "traffic_context",
                    "regex": r"(weekend shopping|holiday anticipation|commuter traffic|normal traffic)",
                    "reason": "Traffic context may be correlated but not causal"
                }
            ]
            print(f"⚠️ Using default spurious patterns for C4")
    
    def check(self, scenario, explanation):
        """
        Check for spurious correlations.
        Uses structured_output.contributing_factors if available.
        """
        # Extract structured data
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            contributing_factors = structured.get('contributing_factors', [])
            explanation_text = explanation.get('explanation', '')
        else:
            structured = {}
            contributing_factors = []
            explanation_text = explanation
        
        violations = []
        confidence = 1.0
        
        # Check structured contributing factors
        for factor in contributing_factors:
            if self._is_spurious_factor(factor, scenario):
                violations.append({
                    'factor': factor,
                    'pattern': 'contributing_factor',
                    'reason': 'Listed as contributing factor but appears spurious'
                })
                confidence *= 0.7
        
        # Check free text patterns
        for pattern in self.patterns:
            matches = re.findall(pattern['regex'], explanation_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = ' '.join([str(part) for part in match if part])
                else:
                    match_text = str(match)
                
                if not self._is_causal_in_scenario(match_text, scenario):
                    violations.append({
                        'factor': match_text,
                        'pattern': pattern['name'],
                        'reason': pattern['reason']
                    })
                    confidence *= 0.7
        
        passed = len(violations) == 0
        used_structured = bool(contributing_factors)
        
        return {
            'checker': 'C4',
            'passed': passed,
            'confidence': round(confidence, 3),
            'reason': 'No spurious correlations detected' if passed else f'{len(violations)} spurious factor(s)',
            'details': {
                'violations': violations,
                'used_structured': used_structured,
                'structured_factors_checked': contributing_factors if used_structured else []
            }
        }
    
    def _is_spurious_factor(self, factor, scenario):
        """Check if a contributing factor is actually spurious"""
        spurious_indicators = ['weather', 'time', 'day', 'color', 'age', 'gender', 
                              'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                              'saturday', 'sunday', 'morning', 'evening', 'afternoon', 'night']
        factor_lower = factor.lower()
        return any(indicator in factor_lower for indicator in spurious_indicators)
    
    def _is_causal_in_scenario(self, factor, scenario):
        """
        Check if a factor is actually causal in the scenario.
        Uses mechanism string and minimal_sufficient_set.
        """
        factor_lower = factor.lower()
        
        # Check mechanism string for causal factors
        mechanism = scenario.get('causal_ground_truth', {}).get('mechanism', '')
        if mechanism:
            # Extract all parts from mechanism (split by arrows)
            parts = re.split(r' → | → |â†’ |â†’', mechanism)
            for part in parts:
                part_lower = part.strip().lower()
                # Check if the factor is part of the mechanism
                if factor_lower in part_lower or part_lower in factor_lower:
                    return True
        
        # Check minimal_sufficient_set
        minimal_set = scenario.get('minimal_sufficient_set', [])
        for causal in minimal_set:
            if isinstance(causal, str):
                causal_lower = causal.lower()
                if factor_lower in causal_lower or causal_lower in factor_lower:
                    return True
        
        # Check primary_cause
        primary_cause = scenario.get('causal_ground_truth', {}).get('primary_cause', '')
        if primary_cause and factor_lower in primary_cause.lower():
            return True
        
        # Check non-causal correlates from ground truth
        non_causal = scenario.get('causal_ground_truth', {}).get('non_causal_correlates', [])
        for nc in non_causal:
            if isinstance(nc, str) and nc.lower() in factor_lower:
                return False
        
        # Common non-causal patterns (fallback)
        common_spurious = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                          'saturday', 'sunday', 'morning', 'evening', 'afternoon', 
                          'night', 'red', 'blue', 'white', 'black', 'silver', 'grey',
                          'yellow', 'green', 'rush hour', 'commuter', 'weekend', 'holiday']
        
        # Check if the factor is purely non-causal
        desc_lower = scenario.get('description', '').lower()
        for word in common_spurious:
            if word in factor_lower:
                # If the word is in the description but not in the mechanism,
                # it's likely non-causal
                if word in desc_lower:
                    if mechanism and word not in mechanism.lower():
                        return False
        
        # If the factor is in the description but not in the mechanism,
        # check if it's explicitly a contributing factor
        if factor_lower in desc_lower and mechanism:
            if factor_lower not in mechanism.lower():
                contributing = scenario.get('causal_ground_truth', {}).get('contributing_factors', [])
                # Check if factor is in contributing factors
                if not any(factor_lower in c.lower() for c in contributing):
                    return False
        
        # Default: treat as causal if we can't determine otherwise
        return True