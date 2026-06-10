# checkers/c1_temporal.py
import re
from datetime import datetime

class C1TemporalChecker:
    def __init__(self):
        self.name = "C₁ Temporal Precedence"
    
    def check(self, scenario, explanation):
        """
        Check if causes precede effects in time.
        Uses structured_output.temporal_sequence if available.
        """
        # Extract structured data if available
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            temporal_sequence = structured.get('temporal_sequence', [])
            explanation_text = explanation.get('explanation', '')
        else:
            structured = {}
            temporal_sequence = []
            explanation_text = explanation
        
        # Use structured temporal_sequence if available
        if temporal_sequence:
            violations = self._check_temporal_sequence(temporal_sequence)
            used_structured = True
        else:
            # Fallback to parsing free text
            violations = self._check_causal_claims(explanation_text, scenario)
            used_structured = False
        
        confidence = 1.0 - (len(violations) * 0.3)
        passed = len(violations) == 0
        
        return {
            'checker': 'C1',
            'passed': passed,
            'confidence': round(max(0.0, confidence), 3),
            'reason': 'All temporal relationships valid' if passed else f'{len(violations)} temporal violation(s)',
            'details': {
                'violations': violations,
                'used_structured': used_structured,
                'claims_found': len(violations) if not used_structured else len(temporal_sequence)
            }
        }
    
    def _check_temporal_sequence(self, sequence):
        """Validate structured temporal sequence"""
        violations = []
        if len(sequence) < 2:
            return violations
        
        for i in range(len(sequence) - 1):
            event_a = sequence[i]
            event_b = sequence[i + 1]
            
            # Extract times if present
            time_a = self._extract_time(event_a)
            time_b = self._extract_time(event_b)
            
            if time_a and time_b and time_a > time_b:
                violations.append({
                    'claim': f"{event_a} → {event_b}",
                    'cause_time': time_a,
                    'effect_time': time_b,
                    'reason': f"Event order reversed"
                })
        return violations
    
    def _extract_time(self, text):
        """Extract time from string like '6:42 PM: event'"""
        match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)?', text)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            meridian = match.group(3)
            if meridian and meridian.upper() == 'PM' and hour < 12:
                hour += 12
            return hour * 60 + minute
        return None
    
    def _check_causal_claims(self, text, scenario):
        """Fallback: extract claims from free text (preserved from original)"""
        claims = self._extract_causal_claims(text)
        timeline = self._build_timeline(scenario)
        
        violations = []
        
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
        
        return violations
    
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
    
    def _build_timeline(self, scenario):
        """Build timeline of events from scenario context"""
        timeline = {}
        
        # Get timeline from scenario if available
        if 'context' in scenario and 'timeline' in scenario['context']:
            for event in scenario['context']['timeline']:
                timeline[event['event']] = event['time']
        
        # Also extract from description
        desc = scenario.get('description', '')
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
    
    def _find_event_time(self, text, timeline):
        """Find if any timeline event is mentioned in text"""
        text_lower = text.lower()
        for event, time in timeline.items():
            if event.lower() in text_lower:
                return time
        return None