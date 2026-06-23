# dashboard.py
import plotly.express as px
import pandas as pd
import sqlite3

def create_dashboard():
    """Generate visualizations from database results."""
    conn = sqlite3.connect('causal_audit.db')
    
    # Load results
    df = pd.read_sql_query("""
        SELECT 
            scenario_id,
            incident_category,
            all_passed,
            json_extract(check_results, '$.C1.passed') as C1,
            json_extract(check_results, '$.C2.passed') as C2,
            json_extract(check_results, '$.C3.passed') as C3,
            json_extract(check_results, '$.C4.passed') as C4,
            json_extract(check_results, '$.C5.passed') as C5
        FROM causal_audit_logs
        ORDER BY created_at DESC
    """, conn)
    
    # Create summary table
    summary = df.groupby('incident_category').agg({
        'all_passed': 'mean',
        'C1': 'mean',
        'C2': 'mean',
        'C3': 'mean',
        'C4': 'mean',
        'C5': 'mean'
    }).round(3) * 100
    
    print("📊 SUMMARY BY CATEGORY")
    print(summary)
    
    return df, summary

if __name__ == "__main__":
    create_dashboard()