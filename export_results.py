# export_results.py
"""
Export validation results in various formats for analysis.
"""

import json
import sqlite3
import pandas as pd
from pathlib import Path

def export_to_csv(db_path='causal_audit.db', output_dir='exports'):
    """Export results to CSV for analysis."""
    Path(output_dir).mkdir(exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    
    # Export main results
    df = pd.read_sql_query("""
        SELECT 
            scenario_id,
            incident_category,
            all_passed,
            created_at,
            run_id
        FROM causal_audit_logs
    """, conn)
    df.to_csv(f'{output_dir}/results.csv', index=False)
    
    # Export detailed check results
    df2 = pd.read_sql_query("""
        SELECT 
            scenario_id,
            check_results
        FROM causal_audit_logs
    """, conn)
    
    # Parse JSON and expand
    rows = []
    for _, row in df2.iterrows():
        checks = json.loads(row['check_results'])
        for name, data in checks.items():
            rows.append({
                'scenario_id': row['scenario_id'],
                'checker': name,
                'passed': data['passed'],
                'confidence': data['confidence'],
                'reason': data['reason']
            })
    
    df_checks = pd.DataFrame(rows)
    df_checks.to_csv(f'{output_dir}/checker_details.csv', index=False)
    
    conn.close()
    print(f"✅ Exported to {output_dir}/")

if __name__ == "__main__":
    export_to_csv()