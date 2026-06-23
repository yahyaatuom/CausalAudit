# checkers/c3_mechanism.py

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class C3MechanismChecker:
    def __init__(self, shared_model=None, confidence_threshold=0.4):
        """
        Initialize C3 Mechanism Checker.
        
        Args:
            shared_model: Sentence transformer model for embeddings
            confidence_threshold: Minimum similarity to pass (lowered from 0.6 to 0.4)
        """
        self.name = "C₃ Mechanism Plausibility"
        self.model = shared_model
        self.confidence_threshold = confidence_threshold  # Lowered for better recall
        
        # Default mechanism templates (fallback if no ground truth)
        self.default_mechanisms = {
            'Weather': ['rain → wet road → reduced traction → collision'],
            'Traffic Accident': ['driver error → vehicle movement → collision'],
            'Road Maintenance': ['hazard → driver reaction → collision'],
            'Public Event': ['crowd → congestion → delay'],
            'Healthcare': ['condition → intervention → outcome'],
            'Finance': ['market event → reaction → consequence']
        }
    
    def check(self, scenario, explanation):
        """
        Check if the mechanism is physically plausible.
        Compares LLM mechanism to ground truth using semantic similarity.
        """
        # Extract structured data
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            llm_mechanism = structured.get('mechanism', '')
            explanation_text = explanation.get('explanation', '')
        else:
            structured = {}
            llm_mechanism = ''
            explanation_text = explanation
        
        # Get ground truth mechanism
        gt_mechanism = scenario.get('causal_ground_truth', {}).get('mechanism', '')
        
        # If no ground truth, use category default
        if not gt_mechanism:
            category = scenario.get('category', 'default')
            default_mechanisms = self.default_mechanisms.get(category, self.default_mechanisms.get('default', ['unknown']))
            gt_mechanism = default_mechanisms[0] if default_mechanisms else ''
        
        # If no LLM mechanism, try to extract from explanation
        if not llm_mechanism and explanation_text:
            llm_mechanism = self._extract_mechanism_from_text(explanation_text)
        
        # --- FIX: Compare LLM mechanism to GROUND TRUTH ---
        if gt_mechanism and llm_mechanism:
            # Calculate semantic similarity
            similarity = self._calculate_similarity(gt_mechanism, llm_mechanism)
            confidence = similarity
            
            # Check if steps match (structural similarity)
            gt_steps = self._parse_steps(gt_mechanism)
            llm_steps = self._parse_steps(llm_mechanism)
            step_match = self._compare_steps(gt_steps, llm_steps)
            
            # Combine scores
            combined_score = (similarity * 0.6) + (step_match * 0.4)
            passed = combined_score >= self.confidence_threshold
            
            return {
                'checker': 'C3',
                'passed': passed,
                'confidence': round(combined_score, 3),
                'reason': f'Mechanism plausibility: {combined_score:.1%} similarity to ground truth',
                'details': {
                    'used_structured': bool(structured),
                    'similarity': similarity,
                    'step_match': step_match,
                    'combined_score': combined_score,
                    'gt_mechanism': gt_mechanism,
                    'llm_mechanism': llm_mechanism,
                    'threshold': self.confidence_threshold
                }
            }
        
        # Fallback: Check physical plausibility
        else:
            return self._check_physical_plausibility(gt_mechanism or llm_mechanism or explanation_text)
    
    def _calculate_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts."""
        if not self.model:
            # Fallback: word overlap
            return self._word_overlap_similarity(text1, text2)
        
        try:
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            return self._word_overlap_similarity(text1, text2)
    
    def _word_overlap_similarity(self, text1, text2):
        """Simple word overlap fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        total = len(words1 | words2)
        return overlap / total if total > 0 else 0.0
    
    def _parse_steps(self, mechanism):
        """Parse mechanism into steps."""
        # Split by arrows
        steps = re.split(r' → | → |â†’ |â†’', mechanism)
        return [s.strip() for s in steps if s.strip()]
    
    def _compare_steps(self, gt_steps, llm_steps):
        """Compare step structures."""
        if not gt_steps or not llm_steps:
            return 0.0
        
        # Check if similar number of steps
        length_similarity = 1.0 - abs(len(gt_steps) - len(llm_steps)) / max(len(gt_steps), len(llm_steps))
        
        # Check key concept overlap
        gt_concepts = set()
        llm_concepts = set()
        
        for step in gt_steps:
            gt_concepts.update(step.lower().split())
        for step in llm_steps:
            llm_concepts.update(step.lower().split())
        
        # Remove common words
        stopwords = {'to', 'the', 'a', 'an', 'of', 'for', 'on', 'at', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'including'}
        gt_concepts = {w for w in gt_concepts if w not in stopwords and len(w) > 2}
        llm_concepts = {w for w in llm_concepts if w not in stopwords and len(w) > 2}
        
        if not gt_concepts or not llm_concepts:
            concept_similarity = 0.0
        else:
            intersection = len(gt_concepts & llm_concepts)
            union = len(gt_concepts | llm_concepts)
            concept_similarity = intersection / union if union > 0 else 0.0
        
        # Combined step similarity
        return (length_similarity * 0.3) + (concept_similarity * 0.7)
    
    def _extract_mechanism_from_text(self, text):
        """Extract mechanism from free text."""
        # Look for mechanism patterns
        patterns = [
            r'"mechanism":\s*"([^"]+)"',
            r'mechanism["\s:]+([^"]+)',
            r'causal\s+chain[:]?\s*([^.]+)',
            r'process[:]?\s*([^.]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no mechanism found, try to extract from explanation
        sentences = text.split('.')
        for sent in sentences:
            if any(word in sent.lower() for word in ['cause', 'lead', 'result', 'because']):
                return sent.strip()
        
        return ''
    
    def _check_physical_plausibility(self, text):
        """Fallback: check if mechanism is physically plausible."""
        # Simple physical plausibility rules
        plausible_indicators = ['rain', 'snow', 'ice', 'collision', 'brake', 'speed', 'impact', 
                               'hydroplane', 'skid', 'rollover', 'rear-end', 'T-bone', 'fire']
        
        text_lower = text.lower()
        plausible = any(indicator in text_lower for indicator in plausible_indicators)
        
        return {
            'checker': 'C3',
            'passed': plausible,
            'confidence': 0.5 if plausible else 0.2,
            'reason': 'Physical plausibility check (fallback)',
            'details': {
                'used_structured': False,
                'fallback': True,
                'plausible': plausible
            }
        }