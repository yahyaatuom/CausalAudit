# checkers/c4_spurious.py

def check(self, scenario, explanation):
    violations = []
    confidence = 1.0
    
    for pattern in self.patterns:
        matches = re.findall(pattern['regex'], explanation, re.IGNORECASE)
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
                confidence *= 0.7  # Penalty per spurious factor
    
    # Reduce confidence if no patterns matched (can't be sure)
    if len(violations) == 0 and len(self.patterns) > 0:
        confidence = 0.8  # Not sure if clean or just missed
    
    passed = len(violations) == 0
    
    return {
        'checker': 'C4',
        'passed': passed,
        'confidence': round(confidence, 3),
        'reason': 'No spurious correlations detected' if passed else f'{len(violations)} spurious factor(s)',
        'details': {'violations': violations, 'confidence_score': confidence}
    }