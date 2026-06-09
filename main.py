# main.py
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

# ============================================================
# CONFIGURATION
# ============================================================

print("🚀 Initializing Causal-Guard Validation Layer...")

shared_model = SentenceTransformer('all-MiniLM-L6-v2')

llm = GroqLLM()
c1_checker = C1TemporalChecker()
c2_checker = C2SpatialChecker()
c3_checker = C3MechanismChecker(shared_model=shared_model)
c4_checker = C4SpuriousChecker()
c5_checker = C5CompletenessChecker()


# ============================================================
# DATABASE HELPER
# ============================================================

def save_to_db(scenario, llm_result, checks):
    """Saves results to PostgreSQL for long-term auditing."""
    try:
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
        print("   💾 Saved to database")
    except Exception as e:
        print(f"⚠️ Database Sync Warning: {e} (Result not saved to SQL)")


# ============================================================
# SCENARIO LOADING WITH GRACEFUL FALLBACK
# ============================================================

def find_scenarios_file():
    """Search for scenarios.json in multiple possible locations"""
    base_dir = os.path.dirname(__file__)
    possible_paths = [
        os.path.join(base_dir, 'data', 'json', 'scenarios.json'),
        os.path.join(base_dir, 'scenarios.json'),
        os.path.join(base_dir, 'data', 'scenarios.json'),
        os.path.join(base_dir, 'json', 'scenarios.json'),
        'data/json/scenarios.json',
        'scenarios.json',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path, False
    
    fallback_path = os.path.join(base_dir, 'data', 'json', 'scenarios.json')
    return fallback_path, True


def create_sample_scenarios_file(file_path):
    """Create a sample scenarios.json file when none exists"""
    print(f"⚠️ No scenarios.json found. Creating sample file at: {file_path}")
    
    sample_scenarios = {
        "metadata": {"generated": "sample", "scenario_count": 2},
        "scenarios": [
            {
                "id": "SAMPLE-01",
                "category": "Weather",
                "complexity_level": 2,
                "description": "Heavy rain caused a delivery van to hydroplane on Sheikh Zayed Road, triggering a five-vehicle chain reaction.",
                "context": {"timeline": [], "locations": [], "environment": {}},
                "causal_ground_truth": {
                    "primary_cause": "Hydroplaning due to standing water",
                    "mechanism": "heavy rain → standing water → tire loses contact → loss of control",
                    "contributing_factors": ["high speed", "insufficient following distance"],
                    "non_causal_correlates": []
                },
                "minimal_sufficient_set": ["heavy_rain", "standing_water", "hydroplaning_physics"]
            },
            {
                "id": "SAMPLE-02",
                "category": "Traffic Accident",
                "complexity_level": 2,
                "description": "A sedan abruptly changed lanes without signaling, causing an SUV to brake hard and get rear-ended by a delivery van.",
                "context": {"timeline": [], "locations": [], "environment": {}},
                "causal_ground_truth": {
                    "primary_cause": "Unsafe lane change without signal",
                    "mechanism": "lane change without check → emergency brake → rear-end collision",
                    "contributing_factors": ["following too closely", "reduced reaction time"],
                    "non_causal_correlates": []
                },
                "minimal_sufficient_set": ["unsafe_lane_change", "no_signal", "insufficient_following_distance"]
            }
        ]
    }
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_scenarios, f, indent=2)
    
    print(f"✅ Created sample file with 2 scenarios")
    print(f"   Replace with your actual 195 scenarios when ready.\n")
    return sample_scenarios


def load_scenarios():
    """Load scenarios with graceful fallback"""
    json_path, is_fallback = find_scenarios_file()
    
    if is_fallback:
        print(f"⚠️ Warning: Using fallback path: {json_path}")
        print(f"   Place your scenarios.json at data/json/scenarios.json\n")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data.get('scenarios', []))} scenarios from {json_path}")
        return data.get('scenarios', [])
        
    except FileNotFoundError:
        print(f"❌ Could not find scenarios.json at any expected location.")
        data = create_sample_scenarios_file(json_path)
        return data.get('scenarios', [])
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        data = create_sample_scenarios_file(json_path)
        return data.get('scenarios', [])


# ============================================================
# SINGLE SCENARIO PROCESSING
# ============================================================

def process_scenario(scenario, index, total):
    """Process a single scenario and return results"""
    print(f"\n{'='*60}")
    print(f"Scenario {index+1}/{total}: {scenario['id']} - {scenario['category']}")
    print(f"{'='*60}")
    
    # Get LLM explanation
    print("🤖 Requesting LLM Analysis...")
    start_time = time.time()
    llm_result = llm.generate_explanation(scenario['description'])
    elapsed = time.time() - start_time
    
    if "Error" in llm_result['explanation']:
        print(f"❌ LLM Error: {llm_result['explanation']}")
        return None
    
    print(f"   Model: {llm_result['model']} ({elapsed:.2f}s)")
    print(f"📝 Explanation: {llm_result['explanation'][:150]}...")
    
    # Run all checkers
    print("🔍 Running Causal-Guard Checks...")
    
    c1 = c1_checker.check(scenario, llm_result['explanation'])
    print(f"   [C1 Temporal]     {'✅ PASS' if c1['passed'] else '❌ FAIL'}")
    
    c2 = c2_checker.check(scenario, llm_result['explanation'])
    print(f"   [C2 Spatial]      {'✅ PASS' if c2['passed'] else '❌ FAIL'}")
    
    c3 = c3_checker.check(scenario, llm_result['explanation'])
    print(f"   [C3 Mechanism]    {'✅ PASS' if c3['passed'] else '❌ FAIL'}")
    if not c3['passed']:
        print(f"      Reason: {c3['reason']}")
    
    c4 = c4_checker.check(scenario, llm_result['explanation'])
    print(f"   [C4 Spurious]     {'✅ PASS' if c4['passed'] else '❌ FAIL'}")
    if not c4['passed']:
        for v in c4['details']['violations']:
            print(f"     - {v['factor']}: {v['reason']}")
    
    c5 = c5_checker.check(scenario, llm_result['explanation'])
    print(f"   [C5 Completeness] {'✅ PASS' if c5['passed'] else '❌ FAIL'}")
    if not c5['passed']:
        print(f"     Missing: {c5['details']['missing']}")
    
    return {
        'scenario_id': scenario['id'],
        'explanation': llm_result['explanation'],
        'checks': {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
        'llm_result': llm_result
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Load scenarios
    scenarios = load_scenarios()
    
    if not scenarios:
        print("❌ No scenarios to process. Exiting.")
        return
    
    print(f"\n📋 Processing {len(scenarios)} scenarios...\n")
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        result = process_scenario(scenario, i, len(scenarios))
        if result:
            results.append(result)
            save_to_db(scenario, result['llm_result'], result['checks'])
        print("-" * 60)
    
    # Print summary
    if results:
        total = len(results)
        c1_p = sum(1 for r in results if r['checks']['C1']['passed'])
        c2_p = sum(1 for r in results if r['checks']['C2']['passed'])
        c3_p = sum(1 for r in results if r['checks']['C3']['passed'])
        c4_p = sum(1 for r in results if r['checks']['C4']['passed'])
        c5_p = sum(1 for r in results if r['checks']['C5']['passed'])
        
        print("\n" + "█"*60)
        print("FINAL VALIDATION SUMMARY")
        print("█"*60)
        print(f"✅ C1 Temporal Consistency:   {c1_p}/{total} ({c1_p/total*100:.1f}%)")
        print(f"📍 C2 Spatial Plausibility:   {c2_p}/{total} ({c2_p/total*100:.1f}%)")
        print(f"🔬 C3 Mechanistic Accuracy:   {c3_p}/{total} ({c3_p/total*100:.1f}%)")
        print(f"🎭 C4 Spurious Correlations:  {c4_p}/{total} ({c4_p/total*100:.1f}%)")
        print(f"📋 C5 Completeness:           {c5_p}/{total} ({c5_p/total*100:.1f}%)")
        
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Full report saved to results.json")
    else:
        print("❌ No valid results to summarize.")


if __name__ == "__main__":
    main()