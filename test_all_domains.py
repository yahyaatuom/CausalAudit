# test_all_domains.py
"""Test C3 on one scenario from each domain."""

from main import (
    ScenarioLoader, load_embedding_model, GroqLLM,
    C3MechanismChecker
)

def test_c3_across_domains():
    """Test C3 on one scenario per domain."""
    
    # Load scenarios
    scenarios = ScenarioLoader.load_scenarios()
    
    # Group by category
    domains = {}
    for s in scenarios:
        cat = s.get('category', 'Unknown')
        if cat not in domains:
            domains[cat] = s
    
    # Load model and checker
    shared_model = load_embedding_model()
    c3 = C3MechanismChecker(shared_model=shared_model)
    llm = GroqLLM()
    
    print("\n" + "="*60)
    print("🔬 TESTING C3 ACROSS ALL DOMAINS")
    print("="*60)
    
    results = {}
    for domain, scenario in domains.items():
        print(f"\n📂 Testing: {domain} - {scenario['id']}")
        
        # Get LLM explanation
        result = llm.generate_explanation(scenario['description'])
        
        # Run C3 check
        c3_result = c3.check(scenario, result['explanation'])
        
        results[domain] = {
            'passed': c3_result['passed'],
            'confidence': c3_result['confidence'],
            'reason': c3_result['reason']
        }
        
        status = "✅ PASS" if c3_result['passed'] else "❌ FAIL"
        print(f"   {status} (conf: {c3_result['confidence']:.2f})")
    
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    for domain, result in results.items():
        status = "✅" if result['passed'] else "❌"
        print(f"{status} {domain}: {result['confidence']:.2f}")
    
    return results

if __name__ == "__main__":
    test_c3_across_domains()