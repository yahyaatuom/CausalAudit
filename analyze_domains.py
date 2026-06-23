# analyze_domains.py
"""
Analyze domain performance from CSV exports.
"""

import csv
import json
import sqlite3
from pathlib import Path

def analyze_from_db(db_path='causal_audit.db'):
    """Analyze performance directly from database."""
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get domain performance
    cur.execute("""
        SELECT 
            incident_category,
            COUNT(*) as total,
            SUM(all_passed) as passed,
            AVG(CASE WHEN all_passed = 1 THEN 1.0 ELSE 0.0 END) * 100 as pass_rate
        FROM causal_audit_logs
        GROUP BY incident_category
        ORDER BY pass_rate DESC
    """)
    
    print("\n" + "="*60)
    print("📊 DOMAIN PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Domain':<20} {'Pass Rate':<12} {'Passed/Total'}")
    print("-"*60)
    
    total_scenarios = 0
    total_passed = 0
    
    for row in cur.fetchall():
        domain, total, passed, rate = row
        total_scenarios += total
        total_passed += passed
        status = "✅" if rate >= 70 else "⚠️" if rate >= 30 else "❌"
        print(f"{status} {domain:<18} {rate:>6.1f}%     {passed:>3}/{total:<3}")
    
    print("-"*60)
    overall_rate = (total_passed / total_scenarios * 100) if total_scenarios > 0 else 0
    print(f"{'OVERALL':<20} {overall_rate:>6.1f}%     {total_passed:>3}/{total_scenarios:<3}")
    
    conn.close()
    
    return {
        'total_scenarios': total_scenarios,
        'total_passed': total_passed,
        'overall_rate': overall_rate
    }

def analyze_checker_failures(db_path='causal_audit.db'):
    """Analyze which checkers are failing most often."""
    
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Get all results and parse check_results JSON
    cur.execute("""
        SELECT scenario_id, incident_category, check_results, all_passed
        FROM causal_audit_logs
        ORDER BY created_at DESC
        LIMIT 100
    """)
    
    checker_failures = {'C1': 0, 'C2': 0, 'C3': 0, 'C4': 0, 'C5': 0}
    total_failures = 0
    
    for row in cur.fetchall():
        scenario_id, category, check_results_json, all_passed = row
        try:
            check_results = json.loads(check_results_json)
            for checker_name, result in check_results.items():
                if not result.get('passed', True):
                    checker_failures[checker_name] = checker_failures.get(checker_name, 0) + 1
                    total_failures += 1
        except:
            pass
    
    conn.close()
    
    print("\n" + "="*60)
    print("🔍 CHECKER FAILURE ANALYSIS")
    print("="*60)
    print(f"{'Checker':<12} {'Failures':<10} {'Failure Rate'}")
    print("-"*60)
    
    for checker, failures in checker_failures.items():
        rate = (failures / total_failures * 100) if total_failures > 0 else 0
        bar = "█" * int(rate / 5)
        print(f"{checker:<12} {failures:<10} {rate:>5.1f}% {bar}")
    
    print("-"*60)
    print(f"Total failures analyzed: {total_failures}")
    
    return checker_failures

def analyze_from_csv(csv_path='exports/checker_details.csv'):
    """Analyze from CSV export if available."""
    
    if not Path(csv_path).exists():
        print(f"⚠️ CSV file not found: {csv_path}")
        print("💡 Run export_results.py first or use database analysis.")
        return None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"\n📊 Analysis from {csv_path}")
    print(f"Total records: {len(rows)}")
    
    # Count checker passes
    checker_passes = {}
    for row in rows:
        checker = row['checker']
        passed = row['passed'] == 'True'
        if checker not in checker_passes:
            checker_passes[checker] = {'pass': 0, 'fail': 0}
        if passed:
            checker_passes[checker]['pass'] += 1
        else:
            checker_passes[checker]['fail'] += 1
    
    print("\n" + "="*60)
    print("📊 CHECKER PERFORMANCE")
    print("="*60)
    print(f"{'Checker':<12} {'Pass Rate':<12} {'Pass/Fail'}")
    print("-"*60)
    
    for checker in ['C1', 'C2', 'C3', 'C4', 'C5']:
        if checker in checker_passes:
            data = checker_passes[checker]
            total = data['pass'] + data['fail']
            rate = data['pass'] / total * 100 if total > 0 else 0
            status = "✅" if rate >= 70 else "⚠️" if rate >= 30 else "❌"
            print(f"{status} {checker:<10} {rate:>6.1f}%     {data['pass']:>3}/{total:<3}")
    
    return checker_passes

if __name__ == "__main__":
    print("\n" + "█"*60)
    print("🔍 CAUSAL-GUARD PERFORMANCE ANALYZER")
    print("█"*60)
    
    # Try database first (more reliable)
    if Path('causal_audit.db').exists():
        summary = analyze_from_db()
        if summary:
            print(f"\n📈 Overall: {summary['total_passed']}/{summary['total_scenarios']} passed ({summary['overall_rate']:.1f}%)")
        
        analyze_checker_failures()
    else:
        # Try CSV as fallback
        analyze_from_csv()
    
    print("\n💡 To generate more detailed reports, run:")
    print("   python export_results.py")