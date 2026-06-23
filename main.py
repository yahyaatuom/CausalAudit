# main.py
"""
Causal-Guard: Neuro-symbolic verification layer for LLM-generated explanations.
Audits causal admissibility against C₁–C₅ constraints.
"""

import json
import time
import sys
import os
import uuid
import re
import sqlite3
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from checkers.c1_temporal import C1TemporalChecker
from checkers.c2_spatial import C2SpatialChecker
from checkers.c3_mechanism import C3MechanismChecker
from checkers.c4_spurious import C4SpuriousChecker
from checkers.c5_completeness import C5CompletenessChecker
from llm_interface import GroqLLM


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    """Configuration settings for Causal-Guard."""
    
    # Database
    DB_PATH = Path(__file__).parent / "causal_audit.db"
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    FALLBACK_EMBEDDING = "paraphrase-MiniLM-L3-v2"
    LLM_MODEL = "llama-3.3-70b-versatile"
    
    # Data settings
    SCENARIOS_PATH = Path(__file__).parent / "data" / "json" / "scenarios.json"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Runtime
    RUN_ID = str(uuid.uuid4())[:8]
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Checker thresholds
    C3_CONFIDENCE_THRESHOLD = 0.6
    C5_COVERAGE_THRESHOLD = 0.5


# ============================================================
# CUSTOM JSON ENCODER
# ============================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ============================================================
# LOGGING
# ============================================================

class Logger:
    """Simple logging utility."""
    
    @staticmethod
    def info(msg: str):
        print(f"ℹ️ {msg}")
    
    @staticmethod
    def success(msg: str):
        print(f"✅ {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"⚠️ {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"❌ {msg}")
    
    @staticmethod
    def debug(msg: str):
        print(f"🔍 {msg}")
    
    @staticmethod
    def section(msg: str, char: str = "=", width: int = 60):
        print(f"\n{char * width}")
        print(f"{msg}")
        print(f"{char * width}")


# ============================================================
# DATABASE
# ============================================================

class Database:
    """SQLite database manager for Causal-Guard."""
    
    def __init__(self, db_path: Path = Config.DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS causal_audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id TEXT,
                incident_category TEXT,
                llm_explanation TEXT,
                check_results TEXT,
                all_passed INTEGER,
                metadata TEXT,
                run_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index for faster queries
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scenario_run ON causal_audit_logs(scenario_id, run_id)")
        
        conn.commit()
        conn.close()
        Logger.success(f"Database initialized at {self.db_path}")
    
    def save_result(self, scenario: Dict, llm_result: Dict, checks: Dict) -> bool:
        """Save a scenario result to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            # Check for duplicate
            cur.execute(
                "SELECT id FROM causal_audit_logs WHERE scenario_id = ? AND run_id = ?",
                (scenario['id'], Config.RUN_ID)
            )
            
            if cur.fetchone():
                Logger.warning(f"Skipping duplicate: {scenario['id']}")
                conn.close()
                return False
            
            all_passed = 1 if all(c['passed'] for c in checks.values()) else 0
            
            checks_json = json.dumps(checks, cls=NumpyEncoder)
            metadata_json = json.dumps(
                {"model": llm_result['model'], "tokens": llm_result['tokens']},
                cls=NumpyEncoder
            )
            
            cur.execute("""
                INSERT INTO causal_audit_logs 
                (scenario_id, incident_category, llm_explanation, check_results, all_passed, metadata, run_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                scenario['id'],
                scenario['category'],
                llm_result['explanation'],
                checks_json,
                all_passed,
                metadata_json,
                Config.RUN_ID
            ))
            
            conn.commit()
            conn.close()
            Logger.success("Saved to database")
            return True
            
        except Exception as e:
            Logger.error(f"Database error: {e}")
            return False


# ============================================================
# EMBEDDING MODEL LOADER
# ============================================================

def load_embedding_model() -> Optional[Any]:
    """Load embedding model with fallback options."""
    Logger.info("Loading embedding model...")
    
    try:
        model = SentenceTransformer(Config.EMBEDDING_MODEL)
        Logger.success(f"Loaded {Config.EMBEDDING_MODEL}")
        return model
    except Exception as e:
        Logger.warning(f"Could not load {Config.EMBEDDING_MODEL}: {e}")
        
        try:
            model = SentenceTransformer(Config.FALLBACK_EMBEDDING)
            Logger.success(f"Loaded {Config.FALLBACK_EMBEDDING}")
            return model
        except Exception as e2:
            Logger.warning(f"Could not load {Config.FALLBACK_EMBEDDING}: {e2}")
            Logger.info("Using TF-IDF fallback for text similarity")
            return None


# ============================================================
# SCENARIO LOADER
# ============================================================

class ScenarioLoader:
    """Load and preprocess scenarios."""
    
    @staticmethod
    def find_scenarios_file() -> Tuple[Path, bool]:
        """Find scenarios.json in multiple locations."""
        base_dir = Path(__file__).parent
        possible_paths = [
            base_dir / "data" / "json" / "scenarios.json",
            base_dir / "scenarios.json",
            base_dir / "data" / "scenarios.json",
            base_dir / "json" / "scenarios.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path, False
        
        return base_dir / "data" / "json" / "scenarios.json", True
    
    @staticmethod
    def create_sample_scenarios(file_path: Path) -> Dict:
        """Create sample scenarios if none exist."""
        Logger.warning(f"No scenarios.json found. Creating sample at: {file_path}")
        
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
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_scenarios, f, indent=2, ensure_ascii=False)
        
        Logger.success(f"Created sample file with 2 scenarios")
        return sample_scenarios
    
    @classmethod
    def load_scenarios(cls) -> List[Dict]:
        """Load scenarios with graceful fallback."""
        json_path, is_fallback = cls.find_scenarios_file()
        
        if is_fallback:
            Logger.warning(f"Using fallback path: {json_path}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            scenarios = data.get('scenarios', [])
            Logger.success(f"Loaded {len(scenarios)} scenarios from {json_path}")
            return scenarios
        except FileNotFoundError:
            Logger.error(f"Could not find scenarios.json at any expected location.")
            data = cls.create_sample_scenarios(json_path)
            return data.get('scenarios', [])
        except json.JSONDecodeError as e:
            Logger.error(f"JSON parsing error: {e}")
            data = cls.create_sample_scenarios(json_path)
            return data.get('scenarios', [])
    
    @staticmethod
    def extract_non_causal_correlates(scenarios: List[Dict]) -> List[Dict]:
        """Extract likely non-causal correlates from descriptions."""
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
                
                for category, terms in common_non_causal.items():
                    for term in terms:
                        if term in desc_lower:
                            non_causal.append(term)
                
                s['causal_ground_truth']['non_causal_correlates'] = non_causal
        
        return scenarios


# ============================================================
# SCENARIO PROCESSING
# ============================================================

class ScenarioProcessor:
    """Process scenarios through the Causal-Guard pipeline."""
    
    def __init__(self, llm: GroqLLM, checkers: Dict):
        self.llm = llm
        self.checkers = checkers
    
    def process(self, scenario: Dict, index: int, total: int) -> Optional[Dict]:
        """Process a single scenario and return results."""
        Logger.section(f"Scenario {index+1}/{total}: {scenario['id']} - {scenario['category']}")
        
        # Get LLM explanation
        Logger.info("Requesting LLM Analysis...")
        start_time = time.time()
        llm_result = self.llm.generate_explanation(scenario['description'])
        elapsed = time.time() - start_time
        
        if "Error" in llm_result['explanation']:
            Logger.error(f"LLM Error: {llm_result['explanation']}")
            return None
        
        Logger.info(f"Model: {llm_result['model']} ({elapsed:.2f}s)")
        Logger.debug(f"Explanation: {llm_result['explanation'][:150]}...")
        
        # Run all checkers
        Logger.info("Running Causal-Guard Checks...")
        results = {}
        
        for name, checker in self.checkers.items():
            result = checker.check(scenario, llm_result['explanation'])
            results[name] = result
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            Logger.info(f"[{name}] {status} (conf: {result['confidence']:.2f})")
        
        return {
            'scenario_id': scenario['id'],
            'explanation': llm_result['explanation'],
            'checks': results,
            'llm_result': llm_result
        }


# ============================================================
# EVALUATION
# ============================================================

class Evaluator:
    """Evaluate Causal-Guard performance."""
    
    def __init__(self, processor: ScenarioProcessor, db: Database):
        self.processor = processor
        self.db = db
    
    def evaluate(self, scenarios: List[Dict], set_name: str) -> List[Dict]:
        """Run evaluation on a set of scenarios."""
        Logger.section(f"EVALUATING ON {set_name.upper()} SET ({len(scenarios)} scenarios)")
        
        results = []
        for i, scenario in enumerate(scenarios):
            result = self.processor.process(scenario, i, len(scenarios))
            if result:
                results.append(result)
                self.db.save_result(scenario, result['llm_result'], result['checks'])
            print("-" * 40)
        
        return results
    
    @staticmethod
    def print_summary(results: List[Dict], scenarios: List[Dict], set_name: str) -> Optional[Dict]:
        """Print summary with accuracy vs ground truth."""
        if not results:
            Logger.warning(f"No results for {set_name} set")
            return None
        
        total = len(results)
        
        # Count PASS rates
        pass_counts = {}
        for checker_name in ['C1', 'C2', 'C3', 'C4', 'C5']:
            pass_counts[checker_name] = sum(1 for r in results if r['checks'][checker_name]['passed'])
        
        # Calculate accuracy vs ground truth
        c1_correct = 0
        c4_correct = 0
        c1_total = 0
        c4_total = 0
        
        for r in results:
            scenario = next((s for s in scenarios if s['id'] == r['scenario_id']), None)
            if not scenario:
                continue
            
            # C1: Check temporal sequence
            mechanism = scenario.get('causal_ground_truth', {}).get('mechanism', '')
            if mechanism:
                c1_total += 1
                steps = re.split(r' → | → |â†’ |â†’', mechanism)
                explanation = r.get('explanation', '').lower()
                key_steps = [s.strip() for s in steps if len(s.strip()) > 5]
                step_found = any(step.lower() in explanation for step in key_steps)
                c1_passed = r['checks']['C1']['passed']
                if c1_passed and step_found:
                    c1_correct += 1
            
            # C4: Check spurious detection
            non_causal = scenario.get('causal_ground_truth', {}).get('non_causal_correlates', [])
            if non_causal:
                c4_total += 1
                c4_passed = r['checks']['C4']['passed']
                violations = r['checks']['C4'].get('details', {}).get('violations', [])
                flagged_any = len(violations) > 0
                correct = (c4_passed and flagged_any) or (not c4_passed and not flagged_any)
                if correct:
                    c4_correct += 1
        
        # Print summary
        Logger.section(f"{set_name.upper()} SET SUMMARY")
        for checker_name, count in pass_counts.items():
            rate = count / total * 100
            print(f"{'✅' if rate > 50 else '⚠️'} {checker_name}: {count}/{total} ({rate:.1f}%)")
        
        if c1_total > 0 or c4_total > 0:
            print(f"\n🎯 ACCURACY VS GROUND TRUTH:")
            if c1_total > 0:
                print(f"   C1 Temporal: {c1_correct}/{c1_total} ({c1_correct/c1_total*100:.1f}%)")
            if c4_total > 0:
                print(f"   C4 Spurious: {c4_correct}/{c4_total} ({c4_correct/c4_total*100:.1f}%)")
        
        return {
            'set_name': set_name,
            'total': total,
            'pass_rates': {k: v/total*100 for k, v in pass_counts.items()},
            'accuracy': {
                'C1': c1_correct/c1_total*100 if c1_total > 0 else None,
                'C4': c4_correct/c4_total*100 if c4_total > 0 else None
            }
        }


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution entry point."""
    Logger.section("🚀 Initializing Causal-Guard Validation Layer")
    
    # Load embedding model
    shared_model = load_embedding_model()
    
    # Initialize LLM
    try:
        llm = GroqLLM()
        Logger.success(f"LLM initialized: {llm.model}")
    except ValueError as e:
        Logger.error(f"Failed to initialize LLM: {e}")
        sys.exit(1)
    
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
    
    # Load scenarios
    scenarios = ScenarioLoader.load_scenarios()
    scenarios = ScenarioLoader.extract_non_causal_correlates(scenarios)
    
    if not scenarios:
        Logger.error("No scenarios to process. Exiting.")
        return
    
    # Split into train and test sets
    Logger.section(f"📊 Total scenarios loaded: {len(scenarios)}")
    train_scenarios, test_scenarios = train_test_split(
        scenarios, 
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=[s.get('category', 'Unknown') for s in scenarios]  # Better stratification
    )
    
    print(f"📚 Training set: {len(train_scenarios)} scenarios ({(1-Config.TEST_SIZE)*100:.0f}%)")
    print(f"🧪 Test set: {len(test_scenarios)} scenarios ({Config.TEST_SIZE*100:.0f}%)")
    print(f"🔑 Run ID: {Config.RUN_ID}")
    print(f"⏰ Timestamp: {Config.TIMESTAMP}\n")
    
    # Initialize processor and evaluator
    processor = ScenarioProcessor(llm, checkers)
    evaluator = Evaluator(processor, db)
    
    # Process training set
    Logger.section("🎯 PROCESSING TRAINING SET", char="█")
    train_results = evaluator.evaluate(train_scenarios, "training")
    train_summary = evaluator.print_summary(train_results, train_scenarios, "training")
    
    # Process test set
    Logger.section("🎯 PROCESSING TEST SET — THIS IS YOUR VALIDATION RESULT", char="█")
    test_results = evaluator.evaluate(test_scenarios, "test")
    test_summary = evaluator.print_summary(test_results, test_scenarios, "test")
    
    # Save results
    train_filename = f"results_train_{Config.TIMESTAMP}_{Config.RUN_ID}.json"
    test_filename = f"results_test_{Config.TIMESTAMP}_{Config.RUN_ID}.json"
    
    with open(train_filename, 'w') as f:
        json.dump(train_results, f, indent=2, cls=NumpyEncoder)
    
    with open(test_filename, 'w') as f:
        json.dump(test_results, f, indent=2, cls=NumpyEncoder)
    
    # Save metadata
    metadata = {
        "run_id": Config.RUN_ID,
        "timestamp": Config.TIMESTAMP,
        "model": llm.model,
        "train_scenarios": len(train_results),
        "test_scenarios": len(test_results),
        "train_summary": train_summary,
        "test_summary": test_summary
    }
    with open(f"metadata_{Config.TIMESTAMP}_{Config.RUN_ID}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    Logger.success(f"Results saved to {train_filename} and {test_filename}")
    
    # Final validation result
    if test_summary:
        Logger.section("🏆 FINAL VALIDATION RESULT (Test Set)", char="█")
        print("This is your actual model performance. Report these numbers.")
        print("Training set numbers are for reference only.")


if __name__ == "__main__":
    main()