# checkers/c3_mechanism.py
from ast import pattern
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class C3MechanismChecker:
    def __init__(self, kb_path='data/mechanism_kb.json', shared_model=None):
        self.name = "C₃ Mechanistic Plausibility"
        
        kb_full_path = Path(__file__).parent.parent / kb_path
        with open(kb_full_path, 'r', encoding='utf-8') as f:
            self.kb = json.load(f)['mechanisms']
        
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.kb_embeddings = self.model.encode([m['description'] for m in self.kb])
        self.similarity_threshold = 0.75  # Increased for better precision
    
    def check(self, scenario, explanation, context=None):
        """
        Check mechanistic plausibility.
        Uses structured_output.mechanism if available.
        """
        # Extract structured data
        if isinstance(explanation, dict):
            structured = explanation.get('structured_output', {})
            mechanism_text = structured.get('mechanism', '')
            primary_cause = structured.get('primary_cause', '')
            llm_confidence = structured.get('confidence', 0.5)
            explanation_text = explanation.get('explanation', '')
        else:
            structured = {}
            mechanism_text = ''
            primary_cause = ''
            llm_confidence = 0.5
            explanation_text = explanation
        
        # Use structured mechanism if available and substantial
        if mechanism_text and len(mechanism_text) > 10:
            return self._check_with_mechanism(mechanism_text, primary_cause, llm_confidence, scenario, context)
        
        # Fallback to semantic search on free text
        return self._check_semantic(explanation_text, scenario, context)
    
    def _check_with_mechanism(self, mechanism_text, primary_cause, llm_confidence, scenario, context):
        """Check using structured mechanism from LLM"""
        category = scenario.get('category', '')
        
        # Adjust threshold based on context
        if context:
            temporal_failed = context.has_violation('C1')
            spatial_failed = context.has_violation('C2')
            if temporal_failed or spatial_failed:
                self.similarity_threshold = 0.55
            else:
                self.similarity_threshold = 0.65
        
        # Domain-specific rule overrides (accept common mechanisms)
        if category == 'Healthcare':
            if any(term in mechanism_text.lower() for term in ['fatigue', 'tired', 'exhaustion']):
                return {
                    'checker': 'C3',
                    'passed': True,
                    'confidence': 0.85,
                    'reason': "Fatigue is a recognized healthcare factor",
                    'details': {'used_structured': True, 'domain_override': True}
                }
            if any(term in mechanism_text.lower() for term in ['training', 'skill', 'experience']):
                return {
                    'checker': 'C3',
                    'passed': True,
                    'confidence': 0.80,
                    'reason': "Training-related factor is valid",
                    'details': {'used_structured': True, 'domain_override': True}
                }
        
        if category == 'Traffic Accident':
            if any(term in mechanism_text.lower() for term in ['speeding', 'excessive speed', 'speed']):
                return {
                    'checker': 'C3',
                    'passed': True,
                    'confidence': 0.80,
                    'reason': "Speeding is a recognized traffic factor",
                    'details': {'used_structured': True, 'domain_override': True}
                }
            if any(term in mechanism_text.lower() for term in ['distraction', 'distracted', 'phone']):
                return {
                    'checker': 'C3',
                    'passed': True,
                    'confidence': 0.85,
                    'reason': "Distraction is a recognized traffic factor",
                    'details': {'used_structured': True, 'domain_override': True}
                }
        
        # Rule-based physical impossibility checks
        if 'black ice' in mechanism_text.lower() or 'black ice' in primary_cause.lower():
            temp = self._extract_temperature(mechanism_text)
            if temp is not None and temp > 0:
                return {
                    'checker': 'C3',
                    'passed': False,
                    'confidence': 0.98,
                    'reason': f"Black ice cannot form at {temp}°C (requires ≤0°C)",
                    'details': {'used_structured': True, 'mechanism': mechanism_text[:200]}
                }
        
        if 'hydroplaning' in mechanism_text.lower():
            if 'standing water' not in mechanism_text.lower() and 'water film' not in mechanism_text.lower():
                return {
                    'checker': 'C3',
                    'passed': False,
                    'confidence': 0.85,
                    'reason': "Hydroplaning requires standing water or water film",
                    'details': {'used_structured': True}
                }
        
        # Semantic verification of mechanism against KB
        mechanism_embedding = self.model.encode([mechanism_text])
        similarities = np.dot(self.kb_embeddings, mechanism_embedding.T).flatten()
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_mech = self.kb[best_idx]
        
        # Boost confidence based on keyword matches
        keyword_matches = sum(1 for kw in best_mech.get('keywords', []) 
                              if kw.lower() in mechanism_text.lower() or kw.lower() in primary_cause.lower())
        
        if keyword_matches >= 3:
            confidence = min(0.95, best_similarity + 0.15)
        elif keyword_matches >= 2:
            confidence = min(0.95, best_similarity + 0.08)
        else:
            confidence = best_similarity * 0.85
        
        # Domain-specific boost
        if category == 'Healthcare' and 'healthcare' in best_mech['name']:
            confidence = min(0.95, confidence + 0.1)
        elif category == 'Traffic Accident' and any(term in best_mech['name'] for term in ['traffic', 'accident', 'collision']):
            confidence = min(0.95, confidence + 0.1)
        
        common_sense_patterns = [
    'rain → wet road', 'fog → low visibility', 'ice → slippery',
    'distraction → delayed reaction', 'speed → loss of control'
]
        for pattern in common_sense_patterns:
            if pattern in mechanism_text.lower():
                return {
            'checker': 'C3',
            'passed': True,
            'confidence': 0.90,
            'reason': f"Common sense mechanism: {pattern}",
            'details': {'used_structured': True, 'common_sense': True}
        }
        if best_similarity < self.similarity_threshold:
            return {
                'checker': 'C3',
                'passed': False,
                'confidence': max(0.1, confidence),
                'reason': f"Mechanism doesn't match known patterns (best match: {best_mech['name']})",
                'details': {'used_structured': True, 'similarity': float(best_similarity)}
            }
        
        return {
            'checker': 'C3',
            'passed': True,
            'confidence': round(confidence, 3),
            'reason': f"Validated via {best_mech['name']}",
            'details': {'used_structured': True, 'similarity': float(best_similarity), 'keyword_matches': keyword_matches}
        }
    
    def _check_semantic(self, text, scenario, context):
        """Fallback: semantic search on free text"""
        mechanism_text = self._extract_mechanism(text)
        
        if not mechanism_text or len(mechanism_text.strip()) < 10:
            return {
                'checker': 'C3',
                'passed': False,
                'confidence': 0.1,
                'reason': "Could not extract mechanism from explanation",
                'details': {'used_structured': False}
            }
        
        category = scenario.get('category', '')
        
        # Domain-specific overrides for fallback
        if category == 'Healthcare':
            if any(term in mechanism_text.lower() for term in ['fatigue', 'tired', 'training', 'skill']):
                return {
                    'checker': 'C3',
                    'passed': True,
                    'confidence': 0.80,
                    'reason': "Recognized healthcare factor",
                    'details': {'used_structured': False, 'domain_override': True}
                }
        
        if category == 'Traffic Accident':
            if any(term in mechanism_text.lower() for term in ['speeding', 'distraction', 'phone', 'fatigue']):
                return {
                    'checker': 'C3',
                    'passed': True,
                    'confidence': 0.80,
                    'reason': "Recognized traffic factor",
                    'details': {'used_structured': False, 'domain_override': True}
                }
        
        # Adjust threshold based on context
        if context:
            temporal_failed = context.has_violation('C1')
            spatial_failed = context.has_violation('C2')
            if temporal_failed or spatial_failed:
                self.similarity_threshold = 0.55
            else:
                self.similarity_threshold = 0.65
        
        # Semantic search
        explanation_embedding = self.model.encode([mechanism_text])
        similarities = np.dot(self.kb_embeddings, explanation_embedding.T).flatten()
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_mech = self.kb[best_idx]
        
        # Rule-based overrides
        if 'black ice' in mechanism_text.lower():
            temp = self._extract_temperature(mechanism_text)
            if temp is not None and temp > 0:
                return {
                    'checker': 'C3',
                    'passed': False,
                    'confidence': 0.95,
                    'reason': f"Black ice cannot form at {temp}°C",
                    'details': {'used_structured': False}
                }
        
        if best_similarity < self.similarity_threshold:
            return {
                'checker': 'C3',
                'passed': False,
                'confidence': max(0.1, 1.0 - best_similarity),
                'reason': f"Unknown mechanism (best match: {best_mech['name']} at {best_similarity:.2f})",
                'details': {'used_structured': False, 'similarity': float(best_similarity)}
            }
        
        return {
            'checker': 'C3',
            'passed': True,
            'confidence': round(float(best_similarity), 3),
            'reason': f"Validated via {best_mech['name']}",
            'details': {'used_structured': False, 'matched': best_mech['name'], 'similarity': float(best_similarity)}
        }
    
    def _extract_mechanism(self, text):
        """Extract the causal mechanism from explanation text"""
        if not text:
            return ""
        
        sentences = text.replace('\n', ' ').split('.')
        mechanism_sentences = []
        causal_markers = ['because', 'due to', 'caused by', 'resulting from', 'triggered by', 'led to', 'resulted in', '→', '->']
        
        for sent in sentences:
            sent_lower = sent.lower()
            if any(marker in sent_lower for marker in causal_markers):
                mechanism_sentences.append(sent.strip())
        
        if mechanism_sentences:
            return '. '.join(mechanism_sentences)
        
        for sent in sentences:
            if ' → ' in sent or ' -> ' in sent:
                mechanism_sentences.append(sent.strip())
        
        if mechanism_sentences:
            return '. '.join(mechanism_sentences)
        
        return text[:300]
    
    def _extract_temperature(self, text):
        """Extract temperature from text"""
        patterns = [
            r'(-?\d+)\s*[°C]',
            r'(-?\d+)\s*C\b',
            r'temperature[s]?\s+of\s+(-?\d+)',
            r'dropped to\s+(-?\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    continue
        return None