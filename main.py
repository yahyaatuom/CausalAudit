import json
import time
import os
from sentence_transformers import SentenceTransformer
from checkers.c1_temporal import C1TemporalChecker
from checkers.c2_spatial import C2SpatialChecker
from checkers.c3_mechanism import C3MechanismChecker
from checkers.c4_spurious import C4SpuriousChecker
from checkers.c5_completeness import C5CompletenessChecker
from llm_interface import GroqLLM
import psycopg2
from psycopg2.extras import Json

print(" Initializing Causal-Guard Validation Layer...")

shared_model = SentenceTransformer('all-MiniLM-L6-v2')

llm = GroqLLM()
c1_checker = C1TemporalChecker() 
c2_checker = C2SpatialChecker()
c3_checker = C3MechanismChecker(shared_model=shared_model)
c4_checker = C4SpuriousChecker()
c5_checker = C5CompletenessChecker()

def save_to_db(scenario, llm_result, checks):
    """Saves results to PostgreSQL for long-term auditing."""
    try:
        # Get password from environment
        DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
        
        conn = psycopg2.connect(
            dbname="causal_guard",
            user="postgres", 
            password=DB_PASSWORD,
            host="localhost"
        )
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS causal_audit_logs (
                id SERIAL PRIMARY KEY,
                scenario_id VARCHAR(50),
                incident_category VARCHAR(50),
                llm_explanation TEXT,
                check_results JSONB,
                all_passed BOOLEAN,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        all_passed = all(c['passed'] for c in checks.values())
        
        insert_query = """
            INSERT INTO causal_audit_logs 
            (scenario_id, incident_category, llm_explanation, check_results, all_passed, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(insert_query, (
            scenario['id'],
            scenario['category'],
            llm_result['explanation'],
            Json(checks),
            all_passed,
            Json({"model": llm_result['model'], "tokens": llm_result['tokens']})
        ))
        conn.commit()
        cur.close()
        conn.close()
        print("  Saved to database")
    except Exception as e:
        print(f" Database Sync Warning: {e} (Result not saved to SQL)")

json_path = os.path.join(os.path.dirname(__file__), 'data', 'json', 'scenarios.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f" Loaded {len(data['scenarios'])} scenarios")
scenarios = data['scenarios']
print(f" Loaded {len(scenarios)} scenarios\n")

results = []

for i, scenario in enumerate(scenarios):
    print(f"\n{'='*60}")
    print(f"Scenario {i+1}/{len(scenarios)}: {scenario['id']} - {scenario['category']}")
    print(f"{'='*60}")
    
    print(" Requesting LLM Analysis...")
    start_time = time.time()
    llm_result = llm.generate_explanation(scenario['description'])
    elapsed = time.time() - start_time
    
    if "Error" in llm_result['explanation']:
        print(f"❌ LLM Error: {llm_result['explanation']}")
        continue

    print(f"   Model: {llm_result['model']} ({elapsed:.2f}s)")
    print(f" Explanation: {llm_result['explanation'][:150]}...")

    print(" Running Causal-Guard Checks...")
    
    c1 = c1_checker.check(scenario, llm_result['explanation'])
    print(f"   [C1 Temporal]  {' PASS' if c1['passed'] else ' FAIL'}")

    c2 = c2_checker.check(scenario, llm_result['explanation'])
    print(f"   [C2 Spatial]   {' PASS' if c2['passed'] else ' FAIL'}")

    c3 = c3_checker.check(scenario, llm_result['explanation'])
    print(f"   [C3 Mechanism] {' PASS' if c3['passed'] else ' FAIL'}")
    if not c3['passed']:
        print(f"      Reason: {c3['reason']}")

    print("\n Running C₄ Spurious Checker...")
    c4 = c4_checker.check(scenario, llm_result['explanation'])
    print(f"   [C4 Spurious]  {' PASS' if c4['passed'] else ' FAIL'}")
    if not c4['passed']:
        for v in c4['details']['violations']:
            print(f"     - {v['factor']}: {v['reason']}")

    print("\n Running C₅ Completeness Checker...")
    c5 = c5_checker.check(scenario, llm_result['explanation'])
    print(f"   [C5 Completeness] {' PASS' if c5['passed'] else ' FAIL'}")
    if not c5['passed']:
        print(f"     Missing: {c5['details']['missing']}")

    checker_suite = {
        "C1": c1, 
        "C2": c2, 
        "C3": c3,
        "C4": c4,
        "C5": c5
    }
    
    results.append({
        'scenario_id': scenario['id'],
        'explanation': llm_result['explanation'],
        'checks': checker_suite
    })
    
    save_to_db(scenario, llm_result, checker_suite)
    print("-" * 60)

total = len(results)
if total > 0:
    c1_p = sum(1 for r in results if r['checks']['C1']['passed'])
    c2_p = sum(1 for r in results if r['checks']['C2']['passed'])
    c3_p = sum(1 for r in results if r['checks']['C3']['passed'])
    c4_p = sum(1 for r in results if r['checks']['C4']['passed'])
    c5_p = sum(1 for r in results if r['checks']['C5']['passed'])

    print("\n" + "█"*60)
    print("FINAL VALIDATION SUMMARY")
    print("█"*60)
    print(f" C1 Temporal Consistency:  {c1_p}/{total} ({c1_p/total*100:.1f}%)")
    print(f" C2 Spatial Plausibility:  {c2_p}/{total} ({c2_p/total*100:.1f}%)")
    print(f" C3 Mechanistic Accuracy:  {c3_p}/{total} ({c3_p/total*100:.1f}%)")
    print(f" C4 Spurious Correlations:  {c4_p}/{total} ({c4_p/total*100:.1f}%)")
    print(f" C5 Completeness:          {c5_p}/{total} ({c5_p/total*100:.1f}%)")
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n Full report saved to results.json")