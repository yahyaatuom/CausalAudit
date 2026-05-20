import re
from datetime import datetime

class C1TemporalChecker:
    def __init__(self):
        self.name = "C₁ Temporal Precedence"
    
    def check(self, scenario, explanation):
        """
        Check if causes precede effects in time
        
        Args:
            scenario: dict from scenarios.json
            explanation: LLM-generated explanation string
        
        Returns:
            dict with passed (bool), reason (str), details (dict)
        """
        
        # Extract causal claims from explanation
        claims = self._extract_causal_claims(explanation)
        
        # Extract timeline from scenario
        timeline = self._extract_timeline(scenario)
        
        violations = []
        
        for claim in claims:
            # Find times for cause and effect
            cause_time = self._find_time(claim['cause'], timeline)
            effect_time = self._find_time(claim['effect'], timeline)
            
            if cause_time and effect_time:
                if cause_time >= effect_time:
                    violations.append({
                        'claim': f"{claim['cause']} → {claim['effect']}",
                        'cause_time': cause_time,
                        'effect_time': effect_time,
                        'reason': f"Cause ({cause_time}) occurs after effect ({effect_time})"
                    })
        
        passed = len(violations) == 0
        
        return {
            'checker': 'C1',
            'passed': passed,
            'reason': 'All causes precede effects' if passed else f'{len(violations)} temporal violation(s)',
            'details': {
                'claims_found': len(claims),
                'violations': violations
            }
        }
    
    def _extract_causal_claims(self, text):
        """Extract cause-effect pairs using patterns"""
        patterns = [
            (r'(.+?)\s+caused\s+(.+)', 'caused'),
            (r'(.+?)\s+led to\s+(.+)', 'led to'),
            (r'(.+?)\s+resulted in\s+(.+)', 'resulted in'),
            (r'due to\s+(.+?),\s+(.+)', 'due to'),
        ]
        
        claims = []
        for pattern, rel in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                claims.append({
                    'cause': match[0].strip(),
                    'effect': match[1].strip(),
                    'relation': rel
                })
        
        return claims
    
    def _extract_timeline(self, scenario):
        """Extract timeline from scenario context"""
        timeline = {}
        
        # Get timeline from scenario if available
        if 'context' in scenario and 'timeline' in scenario['context']:
            for event in scenario['context']['timeline']:
                timeline[event['event']] = event['time']
        
        # Also extract from description
        desc = scenario['description']
        
        # Look for time patterns
        time_pattern = r'(\d{1,2}):(\d{2})\s*(AM|PM)?'
        matches = re.findall(time_pattern, desc)
        
        for i, match in enumerate(matches):
            hour = int(match[0])
            minute = int(match[1])
            meridian = match[2] if len(match) > 2 else ''
            
            if meridian.upper() == 'PM' and hour < 12:
                hour += 12
            
            time_str = f"{hour:02d}:{minute:02d}"
            timeline[f'time_{i}'] = time_str
        
        return timeline
    
    def _find_time(self, text, timeline):
        """Find if any timeline event is mentioned in text"""
        text_lower = text.lower()
        
        for event, time in timeline.items():
            if event.lower() in text_lower:
                return time
        
        return None