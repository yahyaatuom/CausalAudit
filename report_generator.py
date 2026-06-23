# report_generator.py
"""
Generate comprehensive validation reports for Causal-Guard.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    """Generate validation reports from database results."""
    
    def __init__(self, db_path='causal_audit.db'):
        self.db_path = db_path
    
    def generate(self, output_dir='reports'):
        """Generate all reports."""
        Path(output_dir).mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        
        # 1. Overall summary
        self._overall_summary(conn, output_dir)
        
        # 2. Per-domain summary
        self._domain_summary(conn, output_dir)
        
        # 3. Per-checker analysis
        self._checker_analysis(conn, output_dir)
        
        # 4. Failure patterns
        self._failure_patterns(conn, output_dir)
        
        conn.close()
        print(f"📄 Reports saved to {output_dir}/")
    
    def _overall_summary(self, conn, output_dir):
        """Generate overall summary."""
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(all_passed) as passed,
                AVG(CASE WHEN all_passed = 1 THEN 1 ELSE 0 END) * 100 as pass_rate
            FROM causal_audit_logs
        """)
        result = cur.fetchone()
        
        summary = {
            'generated': datetime.now().isoformat(),
            'total_scenarios': result[0],
            'passed_scenarios': result[1],
            'pass_rate': round(result[2], 2)
        }
        
        with open(f'{output_dir}/overall_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"   ✅ Overall: {summary['pass_rate']:.1f}% pass rate")
    
    def _domain_summary(self, conn, output_dir):
        """Generate per-domain summary."""
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                incident_category,
                COUNT(*) as total,
                SUM(all_passed) as passed
            FROM causal_audit_logs
            GROUP BY incident_category
        """)
        
        results = []
        for row in cur.fetchall():
            results.append({
                'domain': row[0],
                'total': row[1],
                'passed': row[2],
                'pass_rate': round(row[2] / row[1] * 100, 2)
            })
        
        with open(f'{output_dir}/domain_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        for r in results:
            print(f"   ✅ {r['domain']}: {r['pass_rate']:.1f}% pass rate")
    
    def _checker_analysis(self, conn, output_dir):
        """Analyze each checker's performance."""
        # This would parse the JSON check_results field
        # Complex but valuable for understanding checker behavior
        pass
    
    def _failure_patterns(self, conn, output_dir):
        """Identify common failure patterns."""
        # Analyze which checkers fail together
        # Identify scenarios that cause multiple failures
        pass


if __name__ == "__main__":
    ReportGenerator().generate()