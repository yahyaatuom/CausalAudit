# checkers/c1_temporal.py
import re
from datetime import datetime

class C1TemporalChecker:
    def __init__(self):
        self.name = "C₁ Temporal Precedence"
    
    def check(self, scenario, explanation):
        """
        Check if causes precede effects in time.
        
        Returns:
            dict with passed (bool), confidence (float), reason (str), details (dict)
        """
        # Extract causal claims
        claims = self._extract_causal_claims(explanation)
        timeline = self._build_timeline(scenario)
        
        violations = []
        confidence = 1.0  # Start with high confidence, reduce per violation
        
        for claim in claims:
            cause_time = self._find_event_time(claim['cause'], timeline)
            effect_time = self._find_event_time(claim['effect'], timeline)
            
            if cause_time and effect_time:
                if cause_time >= effect_time:
                    violations.append({
                        'claim': f"{claim['cause']} → {claim['effect']}",
                        'cause_time': cause_time,
                        'effect_time': effect_time,
                        'reason': f"Cause occurs after effect"
                    })
                    # Reduce confidence based on violation severity
                    confidence *= 0.7
        
        # Adjust confidence based on claim extraction certainty
        if len(claims) == 0:
            confidence *= 0.5  # No claims found → low confidence
        else:
            confidence *= min(1.0, len(claims) / 3.0)  # More claims = higher confidence
        
        passed = len(violations) == 0
        
        return {
            'checker': 'C1',
            'passed': passed,
            'confidence': round(confidence, 3),
            'reason': 'All causes precede effects' if passed else f'{len(violations)} temporal violation(s)',
            'details': {
                'claims_found': len(claims),
                'violations': violations,
                'confidence_factors': {
                    'claims_extracted': len(claims),
                    'violation_penalty': 1 - (confidence / (1.0 if passed else 0.7))
                }
            }
        }
    
    def _extract_causal_claims(self, text):
        # ... existing extraction logic ...
        pass
    
    def _build_timeline(self, scenario):
        # ... existing timeline logic ...
        pass
    
    def _find_event_time(self, text, timeline):
        # ... existing time logic ...
        pass