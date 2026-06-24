# checkers/c4_spurious.py - Enhanced version

class C4SpuriousChecker:
    def check(self, scenario, explanation):
        """
        Check for spurious correlations with ground truth comparison.
        """
        # Get ground truth non-causal correlates
        gt_non_causal = scenario.get('causal_ground_truth', {}).get('non_causal_correlates', [])
        
        # Check if C4 actually flagged any spurious factors
        violations = self._detect_spurious(explanation, scenario)
        flagged_count = len(violations)
        
        # Determine if C4 correctly identified ground truth non-causal factors
        if gt_non_causal:
            # Check if each ground truth non-causal was flagged
            flagged_gt = []
            missing_gt = []
            
            for factor in gt_non_causal:
                if any(factor.lower() in v['factor'].lower() for v in violations):
                    flagged_gt.append(factor)
                else:
                    missing_gt.append(factor)
            
            # C4 passes if it flagged at least one non-causal factor
            # But we track accuracy separately
            passed = len(violations) > 0
            
            return {
                'checker': 'C4',
                'passed': passed,
                'confidence': round(len(flagged_gt) / len(gt_non_causal), 3) if gt_non_causal else 0.5,
                'reason': f'Flagged {len(violations)} spurious factors ({len(flagged_gt)}/{len(gt_non_causal)} ground truth)',
                'details': {
                    'violations': violations,
                    'gt_non_causal': gt_non_causal,
                    'flagged_gt': flagged_gt,
                    'missing_gt': missing_gt,
                    'accuracy_vs_gt': len(flagged_gt) / len(gt_non_causal) if gt_non_causal else 0
                }
            }
        
        # Fallback: original logic
        return self._check_spurious_patterns(explanation, scenario)
    
    def _detect_spurious(self, explanation, scenario):
        """Detect spurious factors in explanation."""
        violations = []
        
        # Common spurious patterns
        spurious_patterns = {
            'day_of_week': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
            'time_of_day': ['morning', 'afternoon', 'evening', 'night', 'rush hour'],
            'vehicle_color': ['red', 'blue', 'white', 'black', 'silver', 'grey'],
            'traffic_context': ['weekend shopping', 'holiday anticipation', 'commuter traffic'],
            'weather_context': ['sunny', 'clear', 'dry', 'warm']
        }
        
        text_lower = str(explanation).lower()
        
        for category, patterns in spurious_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    # Check if this is actually causal
                    if not self._is_causal_in_scenario(pattern, scenario):
                        violations.append({
                            'factor': pattern,
                            'category': category,
                            'reason': f'Potential spurious correlation: {pattern}'
                        })
        
        return violations