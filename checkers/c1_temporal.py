# checkers/c1_temporal.py - Enhanced version

class C1TemporalChecker:
    def check(self, scenario, explanation):
        """
        Check if causes precede effects in time with ground truth comparison.
        """
        # Get ground truth mechanism
        gt_mechanism = scenario.get('causal_ground_truth', {}).get('mechanism', '')
        
        # Extract temporal sequence from explanation
        llm_sequence = self._extract_temporal_sequence(explanation)
        gt_sequence = self._extract_temporal_sequence_from_mechanism(gt_mechanism)
        
        if gt_sequence and llm_sequence:
            # Check if LLM sequence matches ground truth order
            match_score = self._compare_sequences(gt_sequence, llm_sequence)
            accuracy = match_score
            
            # Also check temporal precedence within LLM's own sequence
            violations = self._check_temporal_precedence(llm_sequence)
            
            passed = len(violations) == 0
            confidence = 1.0 - (len(violations) * 0.3)
            
            return {
                'checker': 'C1',
                'passed': passed,
                'confidence': round(confidence, 3),
                'reason': f'No temporal violations (sequence match: {accuracy:.1%})' if passed else f'{len(violations)} violation(s)',
                'details': {
                    'violations': violations,
                    'accuracy_vs_gt': accuracy,
                    'gt_sequence': gt_sequence,
                    'llm_sequence': llm_sequence,
                    'match_score': accuracy
                }
            }
        
        # Fallback to original logic
        return self._check_causal_claims(explanation_text, scenario)
    
    def _extract_temporal_sequence(self, explanation):
        """Extract temporal sequence from structured output."""
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            sequence = structured.get('temporal_sequence', [])
            if sequence:
                return sequence
        
        # Fallback: parse from text
        return self._parse_temporal_events(str(explanation))
    
    def _extract_temporal_sequence_from_mechanism(self, mechanism):
        """Extract events from mechanism string."""
        if not mechanism:
            return []
        
        steps = re.split(r' → | → |â†’ |â†’', mechanism)
        return [s.strip() for s in steps if s.strip()]
    
    def _compare_sequences(self, gt_seq, llm_seq):
        """Compare two temporal sequences for order similarity."""
        if not gt_seq or not llm_seq:
            return 0.0
        
        # Check if key events appear in the same order
        matches = 0
        for i, gt_event in enumerate(gt_seq):
            # Find if this event appears in LLM sequence
            for llm_event in llm_seq:
                if self._semantic_match(gt_event, llm_event):
                    matches += 1
                    break
        
        # Also check order
        order_score = self._check_order(gt_seq, llm_seq)
        
        # Combined score
        return (matches / len(gt_seq) * 0.7) + (order_score * 0.3)
    
    def _semantic_match(self, text1, text2):
        """Check semantic similarity between two texts."""
        # Simple keyword overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total > 0.3 if total > 0 else False
    
    def _check_order(self, gt_seq, llm_seq):
        """Check if LLM sequence preserves ground truth order."""
        gt_indices = {}
        for i, event in enumerate(gt_seq):
            gt_indices[event] = i
        
        llm_order = []
        for event in llm_seq:
            for gt_event in gt_indices:
                if self._semantic_match(event, gt_event):
                    llm_order.append(gt_indices[gt_event])
                    break
        
        # Check if order is preserved
        if len(llm_order) < 2:
            return 0.0
        
        correct_order = 0
        for i in range(len(llm_order) - 1):
            if llm_order[i] < llm_order[i+1]:
                correct_order += 1
        
        return correct_order / (len(llm_order) - 1)