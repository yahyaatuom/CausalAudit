# checkers/c3_mechanism.py - Add this method

def _check_mechanism_plausibility(self, scenario, explanation):
    """
    Enhanced mechanism check with better fallback.
    """
    # Get ground truth and LLM mechanisms
    gt_mechanism = scenario.get('causal_ground_truth', {}).get('mechanism', '')
    llm_mechanism = self._get_llm_mechanism(explanation)
    
    if not gt_mechanism:
        # Generate plausible mechanism from scenario
        gt_mechanism = self._generate_mechanism_from_scenario(scenario)
    
    if gt_mechanism and llm_mechanism:
        # Calculate multiple similarity metrics
        semantic_sim = self._calculate_similarity(gt_mechanism, llm_mechanism)
        keyword_sim = self._keyword_overlap(gt_mechanism, llm_mechanism)
        step_sim = self._step_similarity(gt_mechanism, llm_mechanism)
        
        # Combined confidence
        confidence = (semantic_sim * 0.5) + (keyword_sim * 0.3) + (step_sim * 0.2)
        passed = confidence >= self.confidence_threshold
        
        return {
            'checker': 'C3',
            'passed': passed,
            'confidence': round(confidence, 3),
            'reason': f'Mechanism similarity: {confidence:.1%}',
            'details': {
                'semantic_sim': semantic_sim,
                'keyword_sim': keyword_sim,
                'step_sim': step_sim,
                'gt_mechanism': gt_mechanism,
                'llm_mechanism': llm_mechanism
            }
        }
    
    # Fallback
    return self._check_physical_plausibility(str(explanation))

def _keyword_overlap(self, text1, text2):
    """Calculate keyword overlap similarity."""
    # Extract key terms (remove stopwords)
    stopwords = {'the', 'a', 'an', 'to', 'of', 'for', 'on', 'at', 'with', 'by', 'from'}
    words1 = {w for w in text1.lower().split() if w not in stopwords and len(w) > 2}
    words2 = {w for w in text2.lower().split() if w not in stopwords and len(w) > 2}
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    return intersection / union if union > 0 else 0.0

def _generate_mechanism_from_scenario(self, scenario):
    """Generate a plausible mechanism from scenario data."""
    description = scenario.get('description', '')
    category = scenario.get('category', 'Unknown')
    
    # Extract key phrases
    phrases = self._extract_key_phrases(description)
    
    if phrases:
        return ' → '.join(phrases[:5])  # Use top 5 phrases
    
    # Fallback to domain template
    templates = {
        'Healthcare': 'condition → diagnosis → treatment → outcome',
        'Finance': 'market event → reaction → consequence',
        'Weather': 'weather event → road condition → incident',
        'Traffic Accident': 'driver action → collision → outcome',
        'Road Maintenance': 'hazard → driver response → incident',
        'Public Event': 'crowd → congestion → delay'
    }
    return templates.get(category, 'cause → effect → outcome')