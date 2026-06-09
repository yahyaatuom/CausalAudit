# confusion_matrix.py
import json
import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_PATH = "results.json"


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():
    """Load ground truth from JSON file"""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Ground truth file not found: {GROUND_TRUTH_PATH}")
        print("   Please create data/ground_truth.json with scenario labels")
        sys.exit(1)
    
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ Loaded ground truth for {len(data['scenarios'])} scenarios")
    return data['scenarios']


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results_from_db():
    """Load Causal-Guard results from PostgreSQL"""
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "causal_guard"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost")
        )
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT scenario_id, check_results 
            FROM causal_audit_logs 
            ORDER BY created_at DESC
        """)
        results = cur.fetchall()
        cur.close()
        conn.close()
        
        print(f"✅ Loaded {len(results)} results from database")
        return results
        
    except Exception as e:
        print(f"⚠️ Could not load from database: {e}")
        return load_results_from_json()


def load_results_from_json():
    """Fallback: Load results from results.json"""
    if not os.path.exists(RESULTS_PATH):
        print(f"❌ Results file not found: {RESULTS_PATH}")
        print("   Run main.py first to generate results")
        sys.exit(1)
    
    with open(RESULTS_PATH, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"✅ Loaded {len(results)} results from {RESULTS_PATH}")
    return results


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(ground_truth, predictions):
    """Calculate precision, recall, F1, accuracy for each checker"""
    
    metrics = {
        'C1': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'C2': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'C3': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'C4': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0},
        'C5': {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    }
    
    matched = 0
    
    for pred in predictions:
        scenario_id = pred.get('scenario_id')
        
        if scenario_id not in ground_truth:
            continue
        
        matched += 1
        truth = ground_truth[scenario_id]
        
        # Get predictions from results
        if 'checks' in pred:
            checks = pred['checks']
        elif 'check_results' in pred:
            checks = pred['check_results']
        else:
            continue
        
        for checker in ['C1', 'C2', 'C3', 'C4', 'C5']:
            if checker not in checks:
                continue
            
            actual = truth.get(checker, True)
            pred_passed = checks[checker].get('passed', False) if isinstance(checks[checker], dict) else False
            
            if not actual and not pred_passed:
                metrics[checker]['TP'] += 1
            elif actual and not pred_passed:
                metrics[checker]['FP'] += 1
            elif actual and pred_passed:
                metrics[checker]['TN'] += 1
            elif not actual and pred_passed:
                metrics[checker]['FN'] += 1
    
    return metrics, matched


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_metrics(metrics, total_matched):
    """Print formatted metrics table"""
    
    print("\n" + "="*70)
    print("📊 CAUSAL-GUARD PERFORMANCE METRICS")
    print("="*70)
    
    for checker in ['C1', 'C2', 'C3', 'C4', 'C5']:
        TP = metrics[checker]['TP']
        FP = metrics[checker]['FP']
        TN = metrics[checker]['TN']
        FN = metrics[checker]['FN']
        
        total = TP + FP + TN + FN
        if total == 0:
            continue
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / total if total > 0 else 0
        
        print(f"\n{checker} — {checker.replace('C', 'C₀')} Metrics:")
        print(f"   TP: {TP:2d} | FP: {FP:2d} | TN: {TN:2d} | FN: {FN:2d}")
        print(f"   Precision: {precision:6.1%} | Recall: {recall:6.1%} | F1: {f1:6.1%} | Accuracy: {accuracy:6.1%}")
    
    print("\n" + "="*70)
    print("📋 SUMMARY TABLE FOR PAPER")
    print("="*70)
    
    print("\n| Checker | TP | FP | TN | FN | Precision | Recall | F1 Score | Accuracy |")
    print("|---------|----|----|----|----|-----------|--------|----------|----------|")
    
    for checker in ['C1', 'C2', 'C3', 'C4', 'C5']:
        TP = metrics[checker]['TP']
        FP = metrics[checker]['FP']
        TN = metrics[checker]['TN']
        FN = metrics[checker]['FN']
        
        total = TP + FP + TN + FN
        if total == 0:
            continue
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (TP + TN) / total if total > 0 else 0
        
        print(f"| {checker} | {TP:2d} | {FP:2d} | {TN:2d} | {FN:2d} | {precision:6.1%} | {recall:6.1%} | {f1:6.1%} | {accuracy:6.1%} |")


# ============================================================
# MAIN
# ============================================================

def main():
    print("🔍 Running Causal-Guard Confusion Matrix Analysis...\n")
    
    # Load ground truth
    ground_truth = load_ground_truth()
    
    # Load predictions
    predictions = load_results_from_db()
    
    # Calculate metrics
    metrics, matched = calculate_metrics(ground_truth, predictions)
    
    print(f"\n✅ Matched {matched}/{len(ground_truth)} scenarios")
    
    # Display results
    display_metrics(metrics, matched)
    
    # Save metrics to file
    with open('confusion_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\n💾 Metrics saved to confusion_metrics.json")
    print("📊 Causal-Guard evaluation complete. Use these metrics for paper reporting.")
    print("⚠️ Note: This is an evaluation tool, not a production system.")


if __name__ == "__main__":
    main()