import re
import json
from sentence_transformers import SentenceTransformer
import numpy as np

class C3MechanismChecker:
    def __init__(self, kb_path='data/mechanism_kb.json', shared_model=None):
        self.name = "C₃ Mechanistic Plausibility"
        
        # Load knowledge base
        with open(kb_path, 'r') as f:
            self.kb = json.load(f)['mechanisms']
        
        # Use shared model if provided, otherwise create new one
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Pre-compute embeddings
        self.kb_embeddings = self.model.encode([m['description'] for m in self.kb])
        self.similarity_threshold = 0.6  # Slightly higher for precision

    def check(self, scenario, explanation):
        # 1. Semantic Search for the most relevant mechanism
        mechanism_text = self._extract_mechanism(explanation)
        explanation_embedding = self.model.encode([mechanism_text])
        similarities = np.dot(self.kb_embeddings, explanation_embedding.T).flatten()
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        best_mech = self.kb[best_idx]  # ← This is the correct variable name

        if 'black ice' in explanation.lower():
            temp_match = re.search(r'(\d+)\s*°C',explanation)
            if temp_match:
                temp = int(temp_match.group(1))
                if temp > 0:
                    return {
                        'checker': 'C3',
                        'passed': False,
                        'reason': f"Black ice impossible at {temp}°C (requires ≤0°C)",
                        'details': {'matched': 'black_ice_formation', 'similarity': float(best_similarity)}
                    }
                
        if best_similarity < self.similarity_threshold:
            return {'checker': 'C3', 'passed': False, 'reason': "Unknown mechanism."}

        # 2. Dynamic Condition Verification
        violation = self._evaluate_conditions(best_mech, explanation, scenario)
        
        if violation:
            return {
                'checker': 'C3',
                'passed': False,
                'reason': violation,
                'details': {'matched': best_mech['name'], 'similarity': float(best_similarity)}
            }
        
        # 3. Domain-specific rule-based checks (using best_mech, not best_mechanism)
        if best_mech['name'] == 'hydroplaning':
            if 'standing water' not in explanation.lower() and 'water film' not in explanation.lower():
                return {
                    'checker': 'C3',
                    'passed': False,
                    'reason': "Hydroplaning requires standing water or water film",
                    'details': {'matched': best_mech['name'], 'similarity': float(best_similarity)}
                }
    
        if best_mech['name'] == 'black_ice_formation':
            temp = self._extract_temp(explanation)
            if temp is not None and temp > 0:
                return {
                    'checker': 'C3',
                    'passed': False,
                    'reason': f"Black ice impossible at {temp}°C (requires ≤0°C)",
                    'details': {'matched': best_mech['name'], 'similarity': float(best_similarity)}
                }
    
        if best_mech['name'] == 'dooring':
            if 'cyclist' not in explanation.lower() and 'bike lane' not in explanation.lower():
                return {
                    'checker': 'C3',
                    'passed': False,
                    'reason': "Dooring requires a cyclist in the path",
                    'details': {'matched': best_mech['name'], 'similarity': float(best_similarity)}
                }

        # 4. All checks passed
        return {
            'checker': 'C3',
            'passed': True,
            'reason': f"Validated via {best_mech['name']}",
            'details': {'similarity': float(best_similarity)}
        }

    def _evaluate_conditions(self, mech, explanation, scenario):
        """
        Dynamically checks conditions defined in mechanism_kb.json
        """
        conds = mech.get('conditions', {})
        expl_lower = explanation.lower()
        
        # Check Temperature Logic
        if 'temperature_max' in conds:
            # Look for temp in explanation or scenario context
            current_temp = self._extract_temp(explanation) or scenario.get('context', {}).get('environment', {}).get('temperature')
            if current_temp is not None and current_temp > conds['temperature_max']:
                return f"Physical Law Violation: {mech['name']} cannot occur at {current_temp}°C."

        # Check for Required Keywords (Flexible)
        has_keywords = any(kw.lower() in expl_lower for kw in mech.get('keywords', []))
        if not has_keywords:
            return f"Mechanistic Gap: Explanation for {mech['name']} lacks key physical indicators."

        # Check Invalid Examples (Antipatterns)
        for invalid in mech.get('invalid_examples', []):
            if invalid.lower() in expl_lower:
                return f"Contradiction: Explanation mentions '{invalid}' which invalidates {mech['name']}."

        # Check for Cargo/Vehicle Constraints (for Logistics)
        if conds.get('requires_high_sides') and 'truck' not in expl_lower and 'van' not in expl_lower:
            return "Structural Inconsistency: Mechanism requires a high-sided vehicle."

        return None

    def _extract_temp(self, text):
        match = re.search(r'(-?\d+)\s?[°C]', text)
        return int(match.group(1)) if match else None

    def _extract_mechanism(self, text):
        # Improved extraction: grabs the causal 'middle' of the explanation
        markers = ['because', 'due to', 'resulting in', 'triggered by']
        pattern = "|".join(markers)
        segments = re.split(pattern, text, flags=re.IGNORECASE)
        return " ".join(segments[1:]) if len(segments) > 1 else text[:300]