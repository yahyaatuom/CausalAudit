# main.py
import json
import time
import sys
import os
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
from checkers.c1_temporal import C1TemporalChecker
from checkers.c2_spatial import C2SpatialChecker
from checkers.c3_mechanism import C3MechanismChecker
from checkers.c4_spurious import C4SpuriousChecker
from checkers.c5_completeness import C5CompletenessChecker
from llm_interface import GroqLLM
import psycopg2
from psycopg2.extras import Json
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

print("🚀 Initializing Causal-Guard Validation Layer...")

shared_model = SentenceTransformer('all-MiniLM-L6-v2')
try:
    llm = GroqLLM()
except ValueError as e:
    print(f"\n❌ Failed to initialize LLM: {e}")
    print("Exiting. Please fix the API key issue and try again.\n")
    sys.exit(1)

c1_checker = C1TemporalChecker()
c2_checker = C2SpatialChecker()
c3_checker = C3MechanismChecker(shared_model=shared_model)
c4_checker = C4SpuriousChecker()
c5_checker = C5CompletenessChecker()

# Generate unique identifiers for this run
RUN_ID = str(uuid.uuid4())[:8]
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# main.py - Replace lines 38-175 with this clean version

# ============================================================
# DATABASE (SQLite only - no PostgreSQL dependencies)
# ============================================================

import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(__file__), 'causal_audit.db')

def init_db():
    """Initialize SQLite database with schema"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS causal_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id TEXT,
            incident_category TEXT,
            llm_explanation TEXT,
            check_results TEXT,  -- JSON stored as text
            all_passed INTEGER,  -- 0 or 1
            metadata TEXT,       -- JSON stored as text
            run_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ SQLite database initialized at {DB_PATH}")

def save_to_db(scenario, llm_result, checks):
    """Save results to SQLite"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        
        # Check for duplicate in this run
        cur.execute("""
            SELECT id FROM causal_audit_logs 
            WHERE scenario_id = ? AND run_id = ?
        """, (scenario['id'], RUN_ID))
        
        if cur.fetchone():
            print(f"   ⚠️ Skipping duplicate: {scenario['id']} already in this run")
            conn.close()
            return
        
        all_passed = 1 if all(c['passed'] for c in checks.values()) else 0
        
        cur.execute("""
            INSERT INTO causal_audit_logs 
            (scenario_id, incident_category, llm_explanation, check_results, all_passed, metadata, run_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            scenario['id'],
            scenario['category'],
            llm_result['explanation'],
            json.dumps(checks),
            all_passed,
            json.dumps({"model": llm_result['model'], "tokens": llm_result['tokens']}),
            RUN_ID
        ))
        conn.commit()
        conn.close()
        print("   💾 Saved to database")
        
    except Exception as e:
        print(f"⚠️ Database error: {e}")

# Call this at startup
init_db()
        
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

# Add to main.py after loading scenarios

def extract_non_causal_correlates(scenarios):
    """Extract likely non-causal correlates from descriptions"""
    common_non_causal = {
        'day_of_week': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
        'time_period': ['morning', 'afternoon', 'evening', 'night', 'rush hour'],
        'vehicle_color': ['red', 'blue', 'white', 'black', 'silver', 'grey', 'yellow'],
        'traffic_context': ['weekend shopping', 'holiday anticipation', 'commuter traffic'],
    }
    
    for s in scenarios:
        if 'causal_ground_truth' not in s:
            s['causal_ground_truth'] = {}
        
        if 'non_causal_correlates' not in s['causal_ground_truth']:
            desc_lower = s.get('description', '').lower()
            non_causal = []
            
            # Check for known non-causal patterns in description
            for category, terms in common_non_causal.items():
                for term in terms:
                    if term in desc_lower:
                        non_causal.append(term)
            
            s['causal_ground_truth']['non_causal_correlates'] = non_causal
    
    return scenarios

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
    print(f"   [C1 Temporal]     {'✅ PASS' if c1['passed'] else '❌ FAIL'} (conf: {c1['confidence']:.2f})")
    
    c2 = c2_checker.check(scenario, llm_result['explanation'])
    print(f"   [C2 Spatial]      {'✅ PASS' if c2['passed'] else '❌ FAIL'} (conf: {c2['confidence']:.2f})")
    
    c3 = c3_checker.check(scenario, llm_result['explanation'])
    print(f"   [C3 Mechanism]    {'✅ PASS' if c3['passed'] else '❌ FAIL'} (conf: {c3['confidence']:.2f})")
    
    c4 = c4_checker.check(scenario, llm_result['explanation'])
    print(f"   [C4 Spurious]     {'✅ PASS' if c4['passed'] else '❌ FAIL'} (conf: {c4['confidence']:.2f})")
    
    c5 = c5_checker.check(scenario, llm_result['explanation'])
    print(f"   [C5 Completeness] {'✅ PASS' if c5['passed'] else '❌ FAIL'} (conf: {c5['confidence']:.2f})")
    
    return {
        'scenario_id': scenario['id'],
        'explanation': llm_result['explanation'],
        'checks': {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5},
        'llm_result': llm_result
    }


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================

def split_scenarios(scenarios, test_size=0.2, random_state=42):
    """
    Split scenarios into training and test sets.
    
    Args:
        scenarios: List of scenario dictionaries
        test_size: Proportion for test set (default 0.2 = 20%)
        random_state: For reproducible splits
    
    Returns:
        train_scenarios, test_scenarios
    """
    # Group by base scenario ID (without a/b/c suffix)
    base_groups = {}
    for s in scenarios:
        base_id = s['id'].rstrip('abc')
        if base_id not in base_groups:
            base_groups[base_id] = []
        base_groups[base_id].append(s)
    
    # Split base groups, not individual scenarios
    base_ids = list(base_groups.keys())
    train_ids, test_ids = train_test_split(
        base_ids, test_size=test_size, random_state=random_state
    )
    
    train_scenarios = []
    test_scenarios = []
    
    for bid in train_ids:
        train_scenarios.extend(base_groups[bid])
    for bid in test_ids:
        test_scenarios.extend(base_groups[bid])
    
    return train_scenarios, test_scenarios


def evaluate_on_set(scenarios, set_name):
    """Run evaluation on a set of scenarios"""
    print(f"\n{'='*60}")
    print(f"📊 EVALUATING ON {set_name.upper()} SET ({len(scenarios)} scenarios)")
    print(f"{'='*60}")
    
    results = []
    
    for i, scenario in enumerate(scenarios):
        result = process_scenario(scenario, i, len(scenarios))
        if result:
            results.append(result)
            save_to_db(scenario, result['llm_result'], result['checks'])
        print("-" * 40)
    
    return results


def print_summary(results, set_name):
    """Print summary for a result set"""
    if not results:
        print(f"No results for {set_name} set")
        return None
    
    total = len(results)
    c1_p = sum(1 for r in results if r['checks']['C1']['passed'])
    c2_p = sum(1 for r in results if r['checks']['C2']['passed'])
    c3_p = sum(1 for r in results if r['checks']['C3']['passed'])
    c4_p = sum(1 for r in results if r['checks']['C4']['passed'])
    c5_p = sum(1 for r in results if r['checks']['C5']['passed'])
    
    print(f"\n{'='*60}")
    print(f"📊 {set_name.upper()} SET SUMMARY")
    print(f"{'='*60}")
    print(f"✅ C1 Temporal:      {c1_p}/{total} ({c1_p/total*100:.1f}%)")
    print(f"📍 C2 Spatial:       {c2_p}/{total} ({c2_p/total*100:.1f}%)")
    print(f"🔬 C3 Mechanism:     {c3_p}/{total} ({c3_p/total*100:.1f}%)")
    print(f"🎭 C4 Spurious:      {c4_p}/{total} ({c4_p/total*100:.1f}%)")
    print(f"📋 C5 Completeness:  {c5_p}/{total} ({c5_p/total*100:.1f}%)")
    
    return {
        'set_name': set_name,
        'total': total,
        'C1': c1_p/total*100,
        'C2': c2_p/total*100,
        'C3': c3_p/total*100,
        'C4': c4_p/total*100,
        'C5': c5_p/total*100
    }


# ============================================================
# SAVE RESULTS WITH TIMESTAMP
# ============================================================

def save_results_with_timestamp(results, prefix):
    """Save results with timestamp to avoid overwriting"""
    filename = f"{prefix}_{TIMESTAMP}_{RUN_ID}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to {filename}")
    return filename


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    # Optional: Clear previous results
    
    # Load scenarios
    scenarios = load_scenarios()
    scenarios = extract_non_causal_correlates(scenarios)
    
    if not scenarios:
        print("❌ No scenarios to process. Exiting.")
        return
    
    # Split into train and test sets
    print(f"\n📊 Total scenarios loaded: {len(scenarios)}")
    train_scenarios, test_scenarios = split_scenarios(scenarios, test_size=0.2)
    
    print(f"📚 Training set: {len(train_scenarios)} scenarios (80%)")
    print(f"🧪 Test set: {len(test_scenarios)} scenarios (20%)")
    print(f"   Note: Test set used ONLY for final evaluation, not for tuning.\n")
    print(f"🔑 Run ID: {RUN_ID} (all results tagged with this ID)")
    print(f"⏰ Timestamp: {TIMESTAMP}\n")
    
    # Process training set (for reference)
    print("\n" + "█"*60)
    print("🎯 PROCESSING TRAINING SET (80%)")
    print("█"*60)
    train_results = evaluate_on_set(train_scenarios, "training")
    train_summary = print_summary(train_results, "training")
    
    # Process test set (for final evaluation)
    print("\n" + "█"*60)
    print("🎯 PROCESSING TEST SET (20%) — THIS IS YOUR VALIDATION RESULT")
    print("█"*60)
    test_results = evaluate_on_set(test_scenarios, "test")
    test_summary = print_summary(test_results, "test")
    
    # Save results with timestamps (no overwriting)
    train_filename = save_results_with_timestamp(train_results, "results_train")
    test_filename = save_results_with_timestamp(test_results, "results_test")
    
    # Also save metadata
    metadata = {
        "run_id": RUN_ID,
        "timestamp": TIMESTAMP,
        "model": llm.model,
        "train_scenarios": len(train_results),
        "test_scenarios": len(test_results),
        "train_summary": train_summary,
        "test_summary": test_summary
    }
    with open(f"metadata_{TIMESTAMP}_{RUN_ID}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📄 Training results saved to {train_filename}")
    print(f"📄 Test results saved to {test_filename}")
    print(f"📄 Metadata saved to metadata_{TIMESTAMP}_{RUN_ID}.json")
    print(f"🔑 All results tagged with run_id: {RUN_ID}")
    
    if test_summary:
        print("\n" + "█"*60)
        print("🏆 FINAL VALIDATION RESULT (Test Set)")
        print("█"*60)
        print(f"This is your actual model performance. Report these numbers.")
        print(f"Training set numbers are for reference only (may be optimistic).")


if __name__ == "__main__":
    main()