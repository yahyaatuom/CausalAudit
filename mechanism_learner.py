# mechanism_learner.py
"""
Self-learning mechanism knowledge base.
Extracts, clusters, and suggests causal mechanisms from scenarios.
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional


class MechanismLearner:
    """
    Automatically learns causal mechanisms from scenario data.
    Uses clustering and similarity to build a dynamic knowledge base.
    """
    
    def __init__(self, kb_path='data/mechanism_kb.json'):
        self.kb_path = Path(kb_path)
        self.mechanisms = []
        self.vectorizer = None
        self.embeddings = None
        self._load_kb()
    
    def _load_kb(self):
        """Load existing knowledge base."""
        if self.kb_path.exists():
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.mechanisms = data.get('mechanisms', [])
        else:
            self.mechanisms = []
    
    def save_kb(self):
        """Save knowledge base."""
        with open(self.kb_path, 'w', encoding='utf-8') as f:
            json.dump({'mechanisms': self.mechanisms}, f, indent=2, ensure_ascii=False)
    
    def extract_from_scenario(self, scenario: Dict) -> Dict:
        """
        Extract mechanism from a scenario's ground truth.
        Returns a structured mechanism entry.
        """
        gt = scenario.get('causal_ground_truth', {})
        mechanism = gt.get('mechanism', '')
        category = scenario.get('category', 'Unknown')
        
        if not mechanism:
            return None
        
        # Parse steps
        steps = re.split(r' → | → |â†’ |â†’', mechanism)
        steps = [s.strip() for s in steps if s.strip()]
        
        # Extract keywords
        keywords = []
        for step in steps:
            # Extract meaningful words
            words = re.findall(r'\b[a-zA-Z_]{3,}\b', step)
            keywords.extend([w.lower() for w in words])
        
        # Create entry
        entry = {
            'name': self._generate_name(steps),
            'pattern': mechanism,
            'steps': steps,
            'keywords': list(set(keywords)),
            'category': category,
            'confidence': 0.8,  # Start with high confidence for ground truth
            'source_scenario': scenario.get('id', 'unknown')
        }
        
        return entry
    
    def _generate_name(self, steps: List[str]) -> str:
        """Generate a unique name from steps."""
        if not steps:
            return 'unknown_mechanism'
        
        # Use first and last step for name
        first = steps[0][:20].replace(' ', '_')
        last = steps[-1][:20].replace(' ', '_')
        return f"{first}_to_{last}".lower()
    
    def learn_from_scenarios(self, scenarios: List[Dict]):
        """Extract and learn mechanisms from all scenarios."""
        new_mechanisms = []
        duplicates = 0
        
        for scenario in scenarios:
            entry = self.extract_from_scenario(scenario)
            if entry:
                # Check if this mechanism already exists
                if not self._is_duplicate(entry):
                    new_mechanisms.append(entry)
                else:
                    duplicates += 1
        
        if new_mechanisms:
            self.mechanisms.extend(new_mechanisms)
            self.save_kb()
            self._rebuild_index()
        
        print(f"✅ Learned {len(new_mechanisms)} new mechanisms (skipped {duplicates} duplicates)")
        return new_mechanisms
    
    def _is_duplicate(self, entry: Dict, threshold: float = 0.85) -> bool:
        """Check if a mechanism already exists using similarity."""
        if not self.mechanisms:
            return False
        
        # Simple keyword overlap check
        entry_keywords = set(entry.get('keywords', []))
        for existing in self.mechanisms:
            existing_keywords = set(existing.get('keywords', []))
            if not entry_keywords or not existing_keywords:
                continue
            overlap = len(entry_keywords & existing_keywords) / len(entry_keywords | existing_keywords)
            if overlap > threshold:
                return True
        
        return False
    
    def _rebuild_index(self):
        """Rebuild TF-IDF index for similarity search."""
        if not self.mechanisms:
            return
        
        # Get all mechanism texts
        texts = [m.get('pattern', '') for m in self.mechanisms]
        categories = [m.get('category', 'Unknown') for m in self.mechanisms]
        
        # Build TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.embeddings = self.vectorizer.fit_transform(texts)
        
        print(f"🔍 Rebuilt index with {len(self.mechanisms)} mechanisms")
    
    def find_best_match(self, mechanism_text: str, category: str = None) -> Tuple[Optional[Dict], float]:
        """
        Find the best matching existing mechanism.
        Returns (matched_mechanism, similarity_score).
        """
        if not self.mechanisms or self.vectorizer is None:
            return None, 0.0
        
        # Vectorize the query
        query_vec = self.vectorizer.transform([mechanism_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        # If category specified, boost matching category
        if category:
            for i, m in enumerate(self.mechanisms):
                if m.get('category') == category:
                    similarities[i] *= 1.2  # Boost
        
        # Find best match
        best_idx = similarities.argmax()
        best_score = similarities[best_idx]
        
        if best_score > 0.3:  # Minimum threshold
            return self.mechanisms[best_idx], best_score
        
        return None, best_score
    
    def suggest_mechanism(self, scenario: Dict) -> Dict:
        """
        Suggest a mechanism for a new scenario based on similar ones.
        """
        description = scenario.get('description', '')
        category = scenario.get('category', 'Unknown')
        
        # Try to extract key phrases from description
        key_phrases = self._extract_key_phrases(description)
        
        # Find similar mechanisms
        similar_mechanisms = []
        for m in self.mechanisms:
            if m.get('category') == category:
                # Calculate similarity based on keywords
                if any(k in description.lower() for k in m.get('keywords', [])):
                    similar_mechanisms.append(m)
        
        if similar_mechanisms:
            # Use the most common pattern
            patterns = [m.get('pattern', '') for m in similar_mechanisms]
            from collections import Counter
            pattern_counts = Counter(patterns)
            best_pattern = pattern_counts.most_common(1)[0][0]
            
            return {
                'suggested_mechanism': best_pattern,
                'confidence': min(0.8, 0.5 + 0.3 * (len(similar_mechanisms) / 10)),
                'source_count': len(similar_mechanisms),
                'category': category
            }
        
        return None
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        # Simple extraction: look for noun phrases
        patterns = [
            r'(\w+\s+\w+\s+\w+)',  # 3-word phrases
            r'(\w+\s+\w+)',        # 2-word phrases
        ]
        phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        return phrases[:10]  # Limit to 10


class AutomatedMechanismKB:
    """
    Automated mechanism knowledge base builder.
    Extracts mechanisms from scenarios and builds a searchable index.
    """
    
    def __init__(self):
        self.learner = MechanismLearner()
        self.cache = {}
    
    def build_from_scenarios(self, scenarios: List[Dict]):
        """Build the entire knowledge base from scenarios."""
        print("📚 Building automated mechanism knowledge base...")
        
        # Extract mechanisms from all scenarios
        new_mechanisms = self.learner.learn_from_scenarios(scenarios)
        
        print(f"✅ Knowledge base has {len(self.learner.mechanisms)} total mechanisms")
        
        # Analyze coverage by category
        categories = defaultdict(int)
        for m in self.learner.mechanisms:
            categories[m.get('category', 'Unknown')] += 1
        
        print("\n📊 Category Coverage:")
        for category, count in categories.items():
            print(f"   {category}: {count} mechanisms")
        
        return self.learner.mechanisms
    
    def get_mechanism_for_scenario(self, scenario: Dict) -> Optional[Dict]:
        """
        Get or suggest a mechanism for a scenario.
        First tries to find in ground truth, then suggests based on similarity.
        """
        # Check if scenario has ground truth
        gt = scenario.get('causal_ground_truth', {})
        if gt.get('mechanism'):
            return {
                'mechanism': gt['mechanism'],
                'source': 'ground_truth',
                'confidence': 1.0
            }
        
        # Try to find similar mechanism
        description = scenario.get('description', '')
        category = scenario.get('category', 'Unknown')
        
        # Find best match
        match, score = self.learner.find_best_match(description, category)
        
        if match:
            return {
                'mechanism': match.get('pattern'),
                'source': 'auto_suggestion',
                'confidence': score,
                'matched_mechanism': match.get('name')
            }
        
        # Generate suggestion
        suggestion = self.learner.suggest_mechanism(scenario)
        if suggestion:
            return suggestion
        
        return None
    
    def export_for_checker(self, output_path='data/mechanism_kb_auto.json'):
        """Export the knowledge base in the format expected by C3 checker."""
        export_data = {
            'mechanisms': [],
            'metadata': {
                'total': len(self.learner.mechanisms),
                'generated': 'auto',
                'categories': list(set(m.get('category', 'Unknown') for m in self.learner.mechanisms))
            }
        }
        
        for m in self.learner.mechanisms:
            export_data['mechanisms'].append({
                'name': m.get('name', 'unknown'),
                'pattern': m.get('pattern', ''),
                'category': m.get('category', 'Unknown'),
                'keywords': m.get('keywords', []),
                'confidence': m.get('confidence', 0.7),
                'source': m.get('source_scenario', 'unknown')
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Exported {len(export_data['mechanisms'])} mechanisms to {output_path}")
        return export_data


# ============================================================
# Quick Integration Script
# ============================================================

def build_mechanism_kb():
    """
    Main function to build the automated mechanism knowledge base.
    """
    from main import ScenarioLoader
    
    print("\n" + "█"*60)
    print("🔧 BUILDING AUTOMATED MECHANISM KNOWLEDGE BASE")
    print("█"*60 + "\n")
    
    # Load scenarios
    scenarios = ScenarioLoader.load_scenarios()
    print(f"📂 Loaded {len(scenarios)} scenarios\n")
    
    # Build knowledge base
    kb = AutomatedMechanismKB()
    mechanisms = kb.build_from_scenarios(scenarios)
    
    # Export for checker
    kb.export_for_checker('data/mechanism_kb_auto.json')
    
    # Test on a few scenarios
    print("\n🧪 Testing suggestions on first 5 scenarios:")
    for s in scenarios[:5]:
        result = kb.get_mechanism_for_scenario(s)
        if result:
            print(f"   {s['id']}: {result.get('source')} (conf: {result.get('confidence', 0):.2f})")
    
    return kb


if __name__ == "__main__":
    build_mechanism_kb()