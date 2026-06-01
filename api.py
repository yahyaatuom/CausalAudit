# api.py
import os
import json
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import components from Causal-Guard
from llm_interface import GroqLLM
from checkers.c1_temporal import C1TemporalChecker
from checkers.c2_spatial import C2SpatialChecker
# from checkers.c3_mechanism import C3MechanismChecker  # DISABLED
from checkers.c4_spurious import C4SpuriousChecker
from checkers.c5_completeness import C5CompletenessChecker

# ============================================================
# DATABASE (Optional)
# ============================================================

try:
    import psycopg2
    from psycopg2.extras import Json
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    print("⚠️ PostgreSQL not available. Running without database logging.")

# ============================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================

class ValidateRequest(BaseModel):
    """Request model for validation endpoint"""
    incident: str = Field(..., description="Incident description", min_length=10, max_length=5000)
    explanation: Optional[str] = Field(None, description="Optional pre-generated explanation. If not provided, Causal-Guard generates one.")
    scenario_id: Optional[str] = Field(None, description="Optional scenario ID for tracking")
    model: str = Field("llama-3.3-70b-versatile", description="LLM model to use")

class CheckerResult(BaseModel):
    """Individual checker result"""
    passed: bool
    reason: str
    details: Dict[str, Any] = {}

class ValidateResponse(BaseModel):
    """Response model for validation endpoint"""
    request_id: str
    timestamp: str
    incident: str
    explanation: str
    model_used: str
    admissible: bool
    results: Dict[str, CheckerResult]
    violations: List[Dict[str, Any]]
    latency_ms: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    checkers_available: List[str]

# ============================================================
# LIFESPAN MANAGEMENT (Startup/Shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    print("🚀 Starting Causal-Guard API...")
    app.state.llm = GroqLLM()
    app.state.checkers = {
        'C1': C1TemporalChecker(),
        'C2': C2SpatialChecker(),
        # 'C3': C3MechanismChecker(),
        'C4': C4SpuriousChecker(),
        'C5': C5CompletenessChecker()
    }
    print("✅ Causal-Guard initialized successfully")
    yield
    print("🛑 Shutting down Causal-Guard API...")

# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(
    title="Causal-Guard API",
    description="Neuro-symbolic verification layer for LLM-generated explanations. Audits causal admissibility against C₁–C₅ constraints.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_scenario(incident: str, scenario_id: str = None) -> dict:
    """Build a minimal scenario object for the checkers"""
    return {
        "id": scenario_id or f"api_{uuid.uuid4().hex[:8]}",
        "category": "General",
        "complexity_level": 2,
        "description": incident,
        "context": {
            "timeline": [],
            "locations": [{"name": "unknown", "type": "unknown"}],
            "environment": {}
        },
        "causal_ground_truth": {
            "primary_cause": "",
            "mechanism": "",
            "contributing_factors": [],
            "non_causal_correlates": []
        },
        "minimal_sufficient_set": []
    }


def save_to_db(request_id: str, incident: str, explanation: str, model: str,
               results: dict, admissible: bool, latency_ms: int):
    """Save validation results to PostgreSQL (optional)"""
    if not DB_AVAILABLE:
        return

    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "causal_guard"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            host=os.getenv("DB_HOST", "localhost")
        )
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO validation_logs 
            (request_id, incident, explanation, model, check_results, all_passed, latency_ms, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            request_id, incident, explanation, model,
            Json(results), admissible, latency_ms, datetime.utcnow()
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ DB save failed: {e}")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        checkers_available=list(app.state.checkers.keys())
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
        checkers_available=list(app.state.checkers.keys())
    )


@app.post("/validate", response_model=ValidateResponse)
async def validate(request: ValidateRequest, background_tasks: BackgroundTasks):
    """
    Validate an incident explanation against C₁–C₅ constraints
    """
    request_id = uuid.uuid4().hex[:16]
    start_time = time.time()

    # Input validation
    if len(request.incident) < 10:
        raise HTTPException(status_code=400, detail="Incident too short (min 10 chars)")
    if len(request.incident) > 5000:
        raise HTTPException(status_code=400, detail="Incident too long (max 5000 chars)")

    # Step 1: Build scenario object
    scenario = build_scenario(request.incident, request.scenario_id)

    # Step 2: Get explanation (generate if not provided)
    if request.explanation:
        explanation = request.explanation
        model_used = "user_provided"
    else:
        app.state.llm.model = request.model
        llm_result = app.state.llm.generate_explanation(request.incident)
        explanation = llm_result['explanation']
        model_used = request.model

    # Step 3: Run all checkers
    results = {}
    violations = []

    for checker_id, checker in app.state.checkers.items():
        try:
            result = checker.check(scenario, explanation)
            results[checker_id] = CheckerResult(
                passed=result['passed'],
                reason=result['reason'],
                details=result.get('details', {})
            )
            if not result['passed']:
                violations.append({
                    'constraint': checker_id,
                    'reason': result['reason']
                })
        except Exception as e:
            results[checker_id] = CheckerResult(
                passed=False,
                reason=f"Checker error: {str(e)}",
                details={}
            )
            violations.append({
                'constraint': checker_id,
                'reason': f"Checker error: {str(e)}"
            })

    # Step 4: Determine overall admissibility
    admissible = len(violations) == 0

    # Step 5: Calculate latency
    latency_ms = int((time.time() - start_time) * 1000)

    # Step 6: Save to database (background)
    background_tasks.add_task(
        save_to_db, request_id, request.incident, explanation,
        model_used, results, admissible, latency_ms
    )
    #pass  # DISABLED DB LOGGING FOR NOW

    # Step 7: Return response
    return ValidateResponse(
        request_id=request_id,
        timestamp=datetime.utcnow().isoformat(),
        incident=request.incident,
        explanation=explanation,
        model_used=model_used,
        admissible=admissible,
        results=results,
        violations=violations,
        latency_ms=latency_ms
    )


@app.post("/validate/batch")
async def validate_batch(requests: List[ValidateRequest], background_tasks: BackgroundTasks):
    """
    Batch validate multiple incidents
    """
    results = []
    for req in requests:
        result = await validate(req, background_tasks)
        results.append(result.dict())

    return {
        "total": len(results),
        "results": results
    }


# ============================================================
# RUN WITH: python api.py
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )