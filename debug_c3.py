# debug_c3.py
"""
Debug C3 Mechanism Checker to understand why it's failing.
"""

import json
from main import (
    Config, Logger, ScenarioLoader, load_embedding_model,
    GroqLLM, C3MechanismChecker
)


def debug_c3():
    """Debug C3 checker on a single scenario."""
    Logger.section("🔬 DEBUGGING C3 MECHANISM CHECKER")
    
    # Load embedding model
    shared_model = load_embedding_model()
    
    # Initialize C3 checker
    c3 = C3MechanismChecker(shared_model=shared_model)
    
    # Load scenarios
    scenarios = ScenarioLoader.load_scenarios()
    test_scenario = scenarios[0]  # Use first scenario
    
    print(f"\n📄 Testing on: {test_scenario['id']} - {test_scenario['category']}")
    print(f"Description: {test_scenario['description'][:150]}...")
    print(f"\n📋 Ground Truth Mechanism:")
    print(f"   {test_scenario['causal_ground_truth']['mechanism']}")
    
    # Get LLM explanation
    llm = GroqLLM()
    result = llm.generate_explanation(test_scenario['description'])
    
    print(f"\n🤖 LLM Response (structured):")
    if result.get('structured_output'):
        print(f"   Primary Cause: {result['structured_output'].get('primary_cause', 'N/A')}")
        print(f"   Mechanism: {result['structured_output'].get('mechanism', 'N/A')}")
    
    print(f"\n📝 Raw Explanation (first 300 chars):")
    print(f"   {result['explanation'][:300]}...")
    
    # Run C3 check
    print(f"\n🔍 Running C3 Check...")
    c3_result = c3.check(test_scenario, result['explanation'])
    
    print(f"\n📊 C3 Result:")
    print(f"   Passed: {c3_result['passed']}")
    print(f"   Confidence: {c3_result['confidence']}")
    print(f"   Reason: {c3_result['reason']}")
    print(f"   Details: {json.dumps(c3_result['details'], indent=2)}")


if __name__ == "__main__":
    debug_c3()