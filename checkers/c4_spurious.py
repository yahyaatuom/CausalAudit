# checkers/c4_spurious.py
import re
import json
from pathlib import Path

class C4SpuriousChecker:
    def __init__(self, kb_path='data/spurious_patterns.json'):
        self.name = "C₄ Non-Spuriousness"
        
        kb_path = Path(__file__).parent.parent / kb_path
        with open(kb_path, 'r', encoding='utf-8') as f:
            self.patterns = json.load(f)['patterns']
    
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
        spurious_indicators = ['weather', 'time', 'day', 'color', 'age', 'gender', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'morning', 'evening', 'afternoon', 'night']
        factor_lower = factor.lower()
        return any(indicator in factor_lower for indicator in spurious_indicators)
    
    # c4_spurious.py - Modify _is_causal_in_scenario()

def _is_causal_in_scenario(self, factor, scenario):
    """Check if a factor is actually causal in the scenario"""
    factor_lower = factor.lower()
    
    # Check mechanism string for causal factors
    mechanism = scenario.get('causal_ground_truth', {}).get('mechanism', '')
    if mechanism:
        # Extract all parts from mechanism (split by arrows)
        parts = re.split(r' → | → |â†’ |â†’', mechanism)
        for part in parts:
            if part.strip().lower() in factor_lower:
                return True
    
    # Check minimal_sufficient_set (if populated)
    if 'minimal_sufficient_set' in scenario:
        for causal in scenario['minimal_sufficient_set']:
            if isinstance(causal, str) and causal.lower() in factor_lower:
                return True
    
    # Check if in non_causal_correlates (once populated)
    if 'causal_ground_truth' in scenario:
        non_causal = scenario['causal_ground_truth'].get('non_causal_correlates', [])
        for nc in non_causal:
            if isinstance(nc, str) and nc.lower() in factor_lower:
                return False
    
    # Common non-causal patterns (fallback)
    common_spurious = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 
                      'saturday', 'sunday', 'morning', 'evening', 'afternoon', 
                      'night', 'red', 'blue', 'white', 'black', 'silver', 'grey']
    
    for word in common_spurious:
        if word in factor_lower:
            return False
    
    return True  # Default to treating as causal

    def _is_causal_in_scenario(self, factor, scenario):
        """Check if a factor is actually causal in the scenario"""
        factor_lower = factor.lower()
        
        # Check non-causal correlates from scenario
        if 'causal_ground_truth' in scenario:
            non_causal = scenario['causal_ground_truth'].get('non_causal_correlates', [])
            for nc in non_causal:
                if isinstance(nc, str) and nc.lower() in factor_lower:
                    return False
        
        # Check if it's in minimal sufficient set (then it IS causal)
        if 'minimal_sufficient_set' in scenario:
            for causal in scenario['minimal_sufficient_set']:
                if isinstance(causal, str) and causal.lower() in factor_lower:
                    return True
        
        # Common non-causal patterns
        common_spurious = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'morning', 'evening', 'afternoon', 'night', 'red', 'blue', 'white', 'black', 'silver', 'grey', 'yellow']
        
        for word in common_spurious:
            if word in factor_lower and word not in str(scenario.get('description', '')).lower():
                return False
        
        return True