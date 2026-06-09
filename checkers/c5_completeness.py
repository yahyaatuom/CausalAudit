# checkers/c5_completeness.py

def check(self, scenario, explanation):
    required_factors = scenario.get('minimal_sufficient_set', [])
    
    if not required_factors:
        return {
            'checker': 'C5',
            'passed': True,
            'confidence': 0.5,  # No requirements = low confidence
            'reason': 'No required factors specified',
            'details': {}
        }
    
    mentioned = []
    missing = []
    
    for factor in required_factors:
        if self._factor_mentioned(factor, explanation):
            mentioned.append(factor)
        else:
            missing.append(factor)
    
    coverage = len(mentioned) / len(required_factors)
    passed = coverage >= self.coverage_threshold
    
    # Confidence = coverage score (higher coverage = higher confidence)
    confidence = coverage
    
    # Boost confidence if all core factors present
    core_factors = self._get_core_factors(required_factors, scenario)
    core_missing = [f for f in core_factors if f in missing]
    if len(core_missing) == 0 and not passed:
        confidence = 0.6  # Almost passed but missing secondary factors
    
    return {
        'checker': 'C5',
        'passed': passed,
        'confidence': round(confidence, 3),
        'reason': f'Coverage: {coverage:.1%} ({len(mentioned)}/{len(required_factors)})',
        'details': {
            'required': required_factors,
            'mentioned': mentioned,
            'missing': missing,
            'coverage': coverage,
            'confidence_score': confidence
        }
    }