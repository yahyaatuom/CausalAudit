# checkers/c3_mechanism.py (excerpt showing confidence addition)

def check(self, scenario, explanation):
    # 1. Semantic Search
    mechanism_text = self._extract_mechanism(explanation)
    explanation_embedding = self.model.encode([mechanism_text])
    similarities = np.dot(self.kb_embeddings, explanation_embedding.T).flatten()
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    best_mech = self.kb[best_idx]
    
    # 2. Rule-based overrides (temperature, etc.)
    rule_violation = self._evaluate_conditions(best_mech, explanation, scenario)
    
    if rule_violation:
        return {
            'checker': 'C3',
            'passed': False,
            'confidence': 0.95,  # High confidence — rule violation is certain
            'reason': rule_violation,
            'details': {'matched': best_mech['name'], 'similarity': float(best_similarity)}
        }
    
    # 3. Unknown mechanism
    if best_similarity < self.similarity_threshold:
        return {
            'checker': 'C3',
            'passed': False,
            'confidence': max(0.1, 1.0 - best_similarity),  # Lower confidence for unknown
            'reason': "Unknown mechanism.",
            'details': {'best_similarity': float(best_similarity)}
        }
    
    # 4. Pass with confidence = similarity score
    return {
        'checker': 'C3',
        'passed': True,
        'confidence': round(float(best_similarity), 3),
        'reason': f"Validated via {best_mech['name']}",
        'details': {'matched': best_mech['name'], 'similarity': float(best_similarity)}
    }