# checkers/c3_mechanism.py
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


class C3MechanismChecker:
    def __init__(self, kb_path='data/mechanism_kb.json', shared_model=None):
        self.name = "C₃ Mechanistic Plausibility"
        
        # Load knowledge base
        kb_full_path = Path(__file__).parent.parent / kb_path
        with open(kb_full_path, 'r', encoding='utf-8') as f:
            self.kb = json.load(f)['mechanisms']
        
        # Use shared model if provided, otherwise create new one
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings for knowledge base
        self.kb_embeddings = self.model.encode([m['description'] for m in self.kb])
        self.similarity_threshold = 0.6  # Default threshold
    
    def check(self, scenario, explanation, context=None):
        """
        Check mechanistic plausibility of the explanation.
        
        Args:
            scenario: dict with scenario information
            explanation: str or dict with LLM explanation
            context: Optional CheckerContext for inter-checker communication
        
        Returns:
            dict with passed (bool), confidence (float), reason (str), details (dict)
        """
        
        # Extract explanation text (handle both string and structured output)
        if isinstance(explanation, dict):
            # Structured output from LLM
            explanation_text = explanation.get('explanation', '')
            structured_data = explanation.get('structured_output', {})
            # Prefer structured mechanism if available
            if structured_data and structured_data.get('mechanism'):
                mechanism_text = structured_data['mechanism']
            else:
                mechanism_text = self._extract_mechanism(explanation_text)
        else:
            explanation_text = explanation
            mechanism_text = self._extract_mechanism(explanation_text)
        
        # Adjust threshold based on context (inter-checker communication)
        if context:
            temporal_failed = context.has_violation('C1')
            spatial_failed = context.has_violation('C2')
            
            if temporal_failed or spatial_failed:
                # If timing or location already wrong, mechanism is more suspect
                context.add_note(f"C3: Adjusting threshold due to C1={temporal_failed}, C2={spatial_failed}")
                self.similarity_threshold = 0.5  # Stricter when other checkers failed
            else:
                self.similarity_threshold = 0.6
        
        # Step 1: Rule-based overrides (fast, high precision)
        # Check temperature for black ice
        if 'black ice' in explanation_text.lower() or 'black ice' in mechanism_text.lower():
            temp = self._extract_temperature(explanation_text)
            if temp is not None and temp > 0:
                return {
                    'checker': 'C3',
                    'passed': False,
                    'confidence': 0.98,
                    'reason': f"Black ice cannot form at {temp}°C (requires ≤0°C)",
                    'details': {
                        'rule': 'temperature_threshold',
                        'temperature': temp,
                        'threshold': 0,
                        'matched_mechanism': 'black_ice_formation'
                    }
                }
        
        # Check hydroplaning requires standing water
        if 'hydroplaning' in explanation_text.lower() or 'aquaplaning' in explanation_text.lower():
            if 'standing water' not in explanation_text.lower() and 'water film' not in explanation_text.lower():
                return {
                    'checker': 'C3',
                    'passed': False,
                    'confidence': 0.85,
                    'reason': "Hydroplaning requires standing water or water film on the road surface",
                    'details': {
                        'rule': 'missing_condition',
                        'condition': 'standing_water',
                        'matched_mechanism': 'hydroplaning'
                    }
                }
        
        # Check dooring requires cyclist
        if 'dooring' in explanation_text.lower() or 'car door' in explanation_text.lower():
            if 'cyclist' not in explanation_text.lower() and 'bike lane' not in explanation_text.lower():
                return {
                    'checker': 'C3',
                    'passed': False,
                    'confidence': 0.85,
                    'reason': "Dooring mechanism requires a cyclist in the path",
                    'details': {
                        'rule': 'missing_condition',
                        'condition': 'cyclist',
                        'matched_mechanism': 'dooring'
                    }
                }
        
        # Step 2: Semantic search for mechanism
        if not mechanism_text or len(mechanism_text.strip()) < 10:
            return {
                'checker': 'C3',
                'passed': False,
                'confidence': 0.1,
                'reason': "Could not extract mechanism from explanation",
                'details': {'mechanism_text': mechanism_text[:100] if mechanism_text else 'empty'}
            }
        
        explanation_embedding = self.model.encode([mechanism_text])
        similarities = np.dot(self.kb_embeddings, explanation_embedding.T).flatten()
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_mech = self.kb[best_idx]
        
        # Step 3: Check against known mechanisms
        if best_similarity < self.similarity_threshold:
            return {
                'checker': 'C3',
                'passed': False,
                'confidence': max(0.05, 1.0 - best_similarity),
                'reason': f"Unknown mechanism (best match: {best_mech['name']} at {best_similarity:.2f})",
                'details': {
                    'best_match': best_mech['name'],
                    'similarity': float(best_similarity),
                    'threshold': self.similarity_threshold,
                    'mechanism_text': mechanism_text[:200]
                }
            }
        
        # Step 4: Check conditions from knowledge base
        condition_violation = self._evaluate_conditions(best_mech, explanation_text, scenario)
        if condition_violation:
            return {
                'checker': 'C3',
                'passed': False,
                'confidence': 0.90,
                'reason': condition_violation,
                'details': {
                    'matched': best_mech['name'],
                    'similarity': float(best_similarity),
                    'violated_condition': condition_violation
                }
            }
        
        # Step 5: Pass with confidence based on similarity
        confidence = float(best_similarity)
        
        # Boost confidence if keywords match
        keyword_matches = sum(1 for kw in best_mech.get('keywords', []) 
                              if kw.lower() in explanation_text.lower())
        if keyword_matches >= 2:
            confidence = min(0.95, confidence + 0.1)
        
        return {
            'checker': 'C3',
            'passed': True,
            'confidence': round(confidence, 3),
            'reason': f"Validated via {best_mech['name']} (similarity: {best_similarity:.2f})",
            'details': {
                'matched': best_mech['name'],
                'similarity': float(best_similarity),
                'keywords_matched': keyword_matches,
                'mechanism_text': mechanism_text[:200]
            }
        }
    
    def _evaluate_conditions(self, mech, explanation, scenario):
        """
        Check conditions defined in mechanism_kb.json
        """
        conds = mech.get('conditions', {})
        expl_lower = explanation.lower()
        
        # Check temperature constraints
        if 'temperature_max' in conds:
            current_temp = self._extract_temperature(explanation)
            if current_temp is not None and current_temp > conds['temperature_max']:
                return f"Physical Law Violation: {mech['name']} cannot occur at {current_temp}°C (max {conds['temperature_max']}°C)"
        
        if 'temperature_min' in conds:
            current_temp = self._extract_temperature(explanation)
            if current_temp is not None and current_temp < conds['temperature_min']:
                return f"Physical Law Violation: {mech['name']} requires at least {conds['temperature_min']}°C"
        
        # Check for required keywords
        required_keywords = mech.get('keywords', [])
        if required_keywords:
            has_keyword = any(kw.lower() in expl_lower for kw in required_keywords)
            if not has_keyword:
                return f"Mechanistic Gap: Explanation for {mech['name']} lacks key indicators"
        
        # Check for invalid examples (antipatterns)
        for invalid in mech.get('invalid_examples', []):
            if invalid.lower() in expl_lower:
                return f"Contradiction: Explanation mentions '{invalid}' which invalidates {mech['name']}"
        
        # Check vehicle constraints
        if conds.get('requires_high_sides'):
            if 'truck' not in expl_lower and 'van' not in expl_lower:
                return "Structural Inconsistency: Mechanism requires a high-sided vehicle"
        
        if conds.get('requires_moisture'):
            moisture_terms = ['wet', 'moisture', 'damp', 'rain', 'standing water']
            if not any(term in expl_lower for term in moisture_terms):
                return "Condition Violation: Mechanism requires moisture/water presence"
        
        return None
    
    def _extract_mechanism(self, text):
        """Extract the causal mechanism from explanation text"""
        if not text:
            return ""
        
        # Look for sentences describing how/why
        sentences = text.replace('\n', ' ').split('.')
        mechanism_sentences = []
        
        causal_markers = ['because', 'due to', 'caused by', 'resulting from', 
                         'triggered by', 'led to', 'resulted in', '→', '->']
        
        for sent in sentences:
            sent_lower = sent.lower()
            if any(marker in sent_lower for marker in causal_markers):
                mechanism_sentences.append(sent.strip())
        
        if mechanism_sentences:
            return '. '.join(mechanism_sentences)
        
        # Fallback: look for sentences with causal structure
        for sent in sentences:
            if ' → ' in sent or ' -> ' in sent:
                mechanism_sentences.append(sent.strip())
        
        if mechanism_sentences:
            return '. '.join(mechanism_sentences)
        
        # Last resort: return first 300 characters
        return text[:300]
    
    def _extract_temperature(self, text):
        """Extract temperature from text (supports °C and C)"""
        if not text:
            return None
        
        # Pattern for temperature like "2°C", "5 C", "-2°C"
        patterns = [
            r'(-?\d+)\s*[°C]',      # 2°C, -2°C
            r'(-?\d+)\s*C\b',        # 2 C
            r'temperature[s]?\s+of\s+(-?\d+)',  # temperature of 2
            r'dropped to\s+(-?\d+)',  # dropped to 2
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None
    
    def _extract_temperature_from_scenario(self, scenario):
        """Extract temperature from scenario context if available"""
        context = scenario.get('context', {})
        environment = context.get('environment', {})
        temp = environment.get('temperature')
        if temp is not None:
            return int(temp)
        return None