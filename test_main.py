# test_main.py
"""
Run Causal-Guard on a tiny subset of scenarios for testing.
"""

import json
import sys
import os
from pathlib import Path

# Add the current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from main
from main import (
    Config, Logger, Database, ScenarioLoader, 
    ScenarioProcessor, Evaluator, load_embedding_model,
    GroqLLM, C1TemporalChecker, C2SpatialChecker,
    C3MechanismChecker, C4SpuriousChecker, C5CompletenessChecker
)


def get_test_scenarios(n=5):
    """Get first n scenarios from the full dataset."""
    scenarios = ScenarioLoader.load_scenarios()
    
    if not scenarios:
        Logger.error("No scenarios loaded!")
        return []
    
    # Get first n scenarios
    test_scenarios = scenarios[:n]
    Logger.success(f"Using {len(test_scenarios)} test scenarios")
    
    # Print which scenarios we're using
    for s in test_scenarios:
        print(f"  - {s['id']}: {s['category']}")
    
    return test_scenarios


def run_test():
    """Run the test with a small subset."""
    Logger.section("🧪 RUNNING CAUSAL-GUARD TEST ON SMALL SUBSET")
    
    # Load embedding model
    shared_model = load_embedding_model()
    
    # Initialize LLM
    try:
        llm = GroqLLM()
        Logger.success(f"LLM initialized: {llm.model}")
    except ValueError as e:
        Logger.error(f"Failed to initialize LLM: {e}")
        return
    
    # Initialize checkers
    checkers = {
        'C1': C1TemporalChecker(),
        'C2': C2SpatialChecker(),
        'C3': C3MechanismChecker(shared_model=shared_model),
        'C4': C4SpuriousChecker(),
        'C5': C5CompletenessChecker()
    }
    
    # Initialize database
    db = Database()
    
    # Get test scenarios (only 5)
    scenarios = get_test_scenarios(n=5)
    
    if not scenarios:
        Logger.error("No scenarios to process. Exiting.")
        return
    
    # Initialize processor and evaluator
    processor = ScenarioProcessor(llm, checkers)
    evaluator = Evaluator(processor, db)
    
    # Process only test scenarios
    Logger.section("🎯 PROCESSING TEST SUBSET")
    results = evaluator.evaluate(scenarios, "test_subset")
    
    # Print summary
    evaluator.print_summary(results, scenarios, "test_subset")
    
    # Save results
    import json
    from main import NumpyEncoder
    
    filename = f"test_results_{Config.TIMESTAMP}_{Config.RUN_ID}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    Logger.success(f"Test results saved to {filename}")
    
    # Print quick status
    print("\n" + "="*60)
    print("📊 QUICK STATUS")
    print("="*60)
    if results:
        passed = sum(1 for r in results if all(c['passed'] for c in r['checks'].values()))
        print(f"✅ {passed}/{len(results)} scenarios passed all checks")
    print("="*60)


if __name__ == "__main__":
    run_test()