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
    
    # Cache
    CACHE_DIR = Path(__file__).parent / "cache"
    USE_CACHE = True
    
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
    C3_CONFIDENCE_THRESHOLD = 0.4
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
        
        # Main results table
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
        
        # Checkpoint table for resuming
        cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                set_name TEXT,
                last_scenario_index INTEGER,
                processed_count INTEGER,
                total_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, set_name)
            )
        """)
        
        # Cache table for LLM responses
        cur.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id TEXT,
                description_hash TEXT UNIQUE,
                llm_response TEXT,
                model TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_scenario_run ON causal_audit_logs(scenario_id, run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_run ON evaluation_checkpoints(run_id, set_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cache_hash ON llm_cache(description_hash)")
        
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
    
    def save_checkpoint(self, run_id: str, set_name: str, last_index: int, processed: int, total: int):
        """Save checkpoint for resuming."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT OR REPLACE INTO evaluation_checkpoints 
                (run_id, set_name, last_scenario_index, processed_count, total_count, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                set_name,
                last_index,
                processed,
                total,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            Logger.warning(f"Could not save checkpoint: {e}")
    
    def get_checkpoint(self, run_id: str, set_name: str) -> Optional[Dict]:
        """Get checkpoint for a run and set."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT last_scenario_index, processed_count, total_count
                FROM evaluation_checkpoints
                WHERE run_id = ? AND set_name = ?
            """, (run_id, set_name))
            
            row = cur.fetchone()
            conn.close()
            
            if row:
                return {
                    'last_index': row[0],
                    'processed_count': row[1],
                    'total_count': row[2]
                }
            return None
            
        except Exception as e:
            Logger.warning(f"Could not get checkpoint: {e}")
            return None
    
    def get_cached_response(self, scenario_id: str, description: str) -> Optional[Dict]:
        """Get cached LLM response for a scenario."""
        if not Config.USE_CACHE:
            return None
        
        # Create hash of description
        import hashlib
        description_hash = hashlib.md5(description.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT llm_response, model, created_at
                FROM llm_cache
                WHERE scenario_id = ? OR description_hash = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (scenario_id, description_hash))
            
            row = cur.fetchone()
            conn.close()
            
            if row:
                return {
                    'response': json.loads(row[0]),
                    'model': row[1],
                    'cached_at': row[2]
                }
            return None
            
        except Exception as e:
            Logger.warning(f"Cache retrieval error: {e}")
            return None
    
    def cache_response(self, scenario_id: str, description: str, response: Dict, model: str):
        """Cache an LLM response."""
        if not Config.USE_CACHE:
            return
        
        import hashlib
        description_hash = hashlib.md5(description.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                INSERT OR REPLACE INTO llm_cache 
                (scenario_id, description_hash, llm_response, model, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                scenario_id,
                description_hash,
                json.dumps(response, cls=NumpyEncoder),
                model,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            Logger.debug(f"📦 Cached response for {scenario_id}")
            
        except Exception as e:
            Logger.warning(f"Cache save error: {e}")


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
# SCENARIO PROCESSING WITH CACHING
# ============================================================

class ScenarioProcessor:
    """Process scenarios through the Causal-Guard pipeline with caching."""
    
    def __init__(self, llm: GroqLLM, checkers: Dict, db: Database):
        self.llm = llm
        self.checkers = checkers
        self.db = db
        self.cache_hits = 0
        self.cache_misses = 0
    
    def process(self, scenario: Dict, index: int, total: int) -> Optional[Dict]:
        """Process a single scenario with caching."""
        Logger.section(f"Scenario {index+1}/{total}: {scenario['id']} - {scenario['category']}")
        
        # Check cache first
        cached = self.db.get_cached_response(scenario['id'], scenario['description'])
        if cached:
            self.cache_hits += 1
            Logger.info(f"📦 Cache hit! Using cached response from {cached['cached_at']}")
            llm_result = cached['response']
            
            # Still need to run checkers
            Logger.info("Running Causal-Guard Checks on cached response...")
            checks = self._run_checkers(scenario, llm_result['explanation'])
            
            return {
                'scenario_id': scenario['id'],
                'explanation': llm_result['explanation'],
                'checks': checks,
                'llm_result': llm_result,
                'cached': True
            }
        
        # Cache miss - need to call LLM
        self.cache_misses += 1
        Logger.info("📦 Cache miss - calling LLM...")
        
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
        
        # Cache the response
        self.db.cache_response(scenario['id'], scenario['description'], llm_result, llm_result['model'])
        
        # Run all checkers
        Logger.info("Running Causal-Guard Checks...")
        checks = self._run_checkers(scenario, llm_result['explanation'])
        
        return {
            'scenario_id': scenario['id'],
            'explanation': llm_result['explanation'],
            'checks': checks,
            'llm_result': llm_result,
            'cached': False
        }
    
    def _run_checkers(self, scenario: Dict, explanation: str) -> Dict:
        """Run all checkers on a scenario."""
        results = {}
        for name, checker in self.checkers.items():
            result = checker.check(scenario, explanation)
            results[name] = result
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            Logger.info(f"[{name}] {status} (conf: {result['confidence']:.2f})")
        return results


# ============================================================
# EVALUATION
# ============================================================

class Evaluator:
    """Evaluate Causal-Guard performance with resume capability."""
    
    def __init__(self, processor: ScenarioProcessor, db: Database):
        self.processor = processor
        self.db = db
    
    def evaluate(self, scenarios: List[Dict], set_name: str, 
                 resume: bool = True, run_id: str = None) -> Tuple[List[Dict], int]:
        """
        Run evaluation on a set of scenarios with resume capability.
        
        Returns:
            Tuple of (results, processed_count)
        """
        if run_id is None:
            run_id = Config.RUN_ID
        
        total = len(scenarios)
        start_index = 0
        results = []
        processed_count = 0
        
        # Check LLM status
        status = self.processor.llm.get_status()
        Logger.info(f"LLM Status: {status['model']}, {status['remaining_tokens']} tokens remaining")
        
        # Check if we can process remaining scenarios
        remaining = len(scenarios)
        if not self.processor.llm.can_process(remaining):
            Logger.warning(f"⚠️ Not enough tokens for {remaining} scenarios!")
            Logger.info(f"   Remaining: {status['remaining_tokens']}, Needed: ~{remaining * 600}")
            Logger.info("   Switching to smaller model...")
            self.processor.llm._switch_model('down')
        
        # Check for existing checkpoint
        if resume:
            checkpoint = self.db.get_checkpoint(run_id, set_name)
            if checkpoint:
                start_index = checkpoint['last_index'] + 1
                processed_count = checkpoint['processed_count']
                Logger.info(f"🔄 Resuming from scenario {start_index+1}/{total} (already processed {processed_count})")
        
        if start_index >= total:
            Logger.info(f"✅ All {total} scenarios already processed for {set_name}")
            return [], total
        
        Logger.section(f"EVALUATING ON {set_name.upper()} SET ({total - start_index} remaining of {total})")
        
        for i in range(start_index, total):
            scenario = scenarios[i]
            result = self.processor.process(scenario, i, total)
            
            if result:
                results.append(result)
                self.db.save_result(scenario, result['llm_result'], result['checks'])
                processed_count += 1
                
                # Save checkpoint after every 5 scenarios
                if processed_count % 5 == 0:
                    self.db.save_checkpoint(run_id, set_name, i, processed_count, total)
                    Logger.info(f"💾 Checkpoint saved: {processed_count}/{total} scenarios processed")
                    
                    # Check token status periodically
                    status = self.processor.llm.get_status()
                    if status['remaining_tokens'] < 10000:
                        Logger.warning(f"⚠️ Low tokens: {status['remaining_tokens']} remaining")
                        if len(self.processor.llm.api_keys) > 1:
                            self.processor.llm._rotate_key()
            
            print("-" * 40)
        
        # Save final checkpoint
        self.db.save_checkpoint(run_id, set_name, total - 1, processed_count, total)
        Logger.success(f"✅ Completed {processed_count} scenarios for {set_name} set")
        
        # Print cache stats
        Logger.info(f"📊 Cache stats: {self.processor.cache_hits} hits, {self.processor.cache_misses} misses")
        
        return results, processed_count
    
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
    """Main execution entry point with resume capability."""
    Logger.section("🚀 Initializing Causal-Guard Validation Layer")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Causal-Guard Validation")
    parser.add_argument('--no-resume', action='store_true', help='Disable resume functionality')
    parser.add_argument('--run-id', type=str, help='Specific run ID to resume')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--model', type=str, help='Specify LLM model to use')
    args = parser.parse_args()
    
    # Apply settings
    if args.no_cache:
        Config.USE_CACHE = False
        Logger.warning("Cache disabled")
    
    # Load embedding model
    shared_model = load_embedding_model()
    
    # Initialize LLM with optional model
    try:
        model = args.model or Config.LLM_MODEL
        llm = GroqLLM(model=model)
        Logger.success(f"LLM initialized: {llm.current_model}")
        
        # Show LLM status
        status = llm.get_status()
        print(f"   API Keys: {status['api_keys']}")
        print(f"   Tokens remaining: {status['remaining_tokens']}")
        print(f"   Can process 100 scenarios: {status['can_process_100']}")
        
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
        stratify=[s.get('category', 'Unknown') for s in scenarios]
    )
    
    print(f"📚 Training set: {len(train_scenarios)} scenarios ({(1-Config.TEST_SIZE)*100:.0f}%)")
    print(f"🧪 Test set: {len(test_scenarios)} scenarios ({Config.TEST_SIZE*100:.0f}%)")
    
    # Use provided run_id or generate new one
    run_id = args.run_id if args.run_id else Config.RUN_ID
    print(f"🔑 Run ID: {run_id}")
    print(f"⏰ Timestamp: {Config.TIMESTAMP}")
    print(f"🔄 Resume: {'Disabled' if args.no_resume else 'Enabled'}")
    print(f"📦 Cache: {'Enabled' if Config.USE_CACHE else 'Disabled'}\n")
    
    # Initialize processor and evaluator
    processor = ScenarioProcessor(llm, checkers, db)
    evaluator = Evaluator(processor, db)
    
    # Check if training set is already complete
    train_checkpoint = db.get_checkpoint(run_id, "training")
    if train_checkpoint and train_checkpoint['processed_count'] >= len(train_scenarios):
        Logger.success(f"✅ Training set already complete ({train_checkpoint['processed_count']} scenarios)")
        train_results = []
        train_summary = None
    else:
        # Process training set with resume
        Logger.section("🎯 PROCESSING TRAINING SET", char="█")
        train_results, train_processed = evaluator.evaluate(
            train_scenarios, "training", 
            resume=not args.no_resume,
            run_id=run_id
        )
        train_summary = evaluator.print_summary(train_results, train_scenarios, "training")
    
    # Check if test set is already complete
    test_checkpoint = db.get_checkpoint(run_id, "test")
    if test_checkpoint and test_checkpoint['processed_count'] >= len(test_scenarios):
        Logger.success(f"✅ Test set already complete ({test_checkpoint['processed_count']} scenarios)")
        test_results = []
        test_summary = None
    else:
        # Process test set with resume
        Logger.section("🎯 PROCESSING TEST SET — THIS IS YOUR VALIDATION RESULT", char="█")
        test_results, test_processed = evaluator.evaluate(
            test_scenarios, "test",
            resume=not args.no_resume,
            run_id=run_id
        )
        test_summary = evaluator.print_summary(test_results, test_scenarios, "test")
    
    # Save results (only if we have new results)
    if train_results:
        train_filename = f"results_train_{Config.TIMESTAMP}_{run_id}.json"
        with open(train_filename, 'w') as f:
            json.dump(train_results, f, indent=2, cls=NumpyEncoder)
        Logger.success(f"Training results saved to {train_filename}")
    
    if test_results:
        test_filename = f"results_test_{Config.TIMESTAMP}_{run_id}.json"
        with open(test_filename, 'w') as f:
            json.dump(test_results, f, indent=2, cls=NumpyEncoder)
        Logger.success(f"Test results saved to {test_filename}")
    
    # Save metadata
    metadata = {
        "run_id": run_id,
        "timestamp": Config.TIMESTAMP,
        "model": llm.current_model,
        "train_scenarios": len(train_scenarios),
        "test_scenarios": len(test_scenarios),
        "train_summary": train_summary,
        "test_summary": test_summary,
        "resume_enabled": not args.no_resume,
        "cache_enabled": Config.USE_CACHE,
        "llm_status": llm.get_status()
    }
    with open(f"metadata_{Config.TIMESTAMP}_{run_id}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    Logger.success(f"Metadata saved to metadata_{Config.TIMESTAMP}_{run_id}.json")
    
    # Final validation result
    if test_summary:
        Logger.section("🏆 FINAL VALIDATION RESULT (Test Set)", char="█")
        print("This is your actual model performance. Report these numbers.")
        print("Training set numbers are for reference only.")
    
    # Print resume info
    print(f"\n💡 To resume this run later: python main.py --run-id {run_id}")
    print(f"💡 To start fresh: python main.py --no-resume")
    print(f"💡 To use a different model: python main.py --model llama-3.1-8b-instant")
    print(f"💡 To disable cache: python main.py --no-cache")


if __name__ == "__main__":
    main()