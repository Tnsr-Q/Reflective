from fastapi import FastAPI, APIRouter, HTTPException, Depends, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (MUST use env)
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# App + Router (all backend APIs MUST be under /api)
app = FastAPI(title="AI-Reflect MVP")
api = APIRouter(prefix="/api")

# ----- Pydantic Schemas -----
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

ProjectStatus = Literal['draft', 'live', 'archived']
class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default="", max_length=5000)
    status: ProjectStatus = 'draft'

class Project(ProjectCreate):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProjectUpdate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=5000)
    status: Optional[ProjectStatus] = None

class ReflectionCreate(BaseModel):
    project_id: str
    # Either provide text, or provide prompt to generate with AI
    text: Optional[str] = None
    prompt: Optional[str] = None

class Reflection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    text: str
    vector: List[float] = Field(default_factory=list)
    cluster: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

Severity = Literal['low', 'medium', 'high', 'critical']
class AnomalyCreate(BaseModel):
    project_id: str
    detail: str = Field(min_length=1, max_length=2000)
    severity: Severity = 'low'

class Anomaly(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    detail: str
    severity: Severity
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)
    size: Optional[str] = Field(default="512x512")
    quality: Optional[str] = Field(default="standard")

class ImageResponse(BaseModel):
    url: str

class AIHealth(BaseModel):
    ok: bool
    reason: Optional[str] = None

# ----- OpenAI (via Emergent LLM key from env) -----
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')
OPENAI_EMBED_MODEL = os.environ.get('OPENAI_EMBED_MODEL', 'text-embedding-3-small')

# Lazy import to avoid import error if not installed yet
async_openai_client = None

def get_openai_client():
    global async_openai_client
    if async_openai_client is None:
        try:
            from openai import AsyncOpenAI  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"OpenAI SDK not installed: {e}")
        if not OPENAI_KEY:
            raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY in backend environment")
        async_openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
    return async_openai_client

# ----- Utilities -----
async def serialize_dt(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    for k, v in list(out.items()):
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    return out

# Simple deterministic pseudo-embedding as last-resort fallback
import hashlib

def fallback_embed(text: str, dim: int = 64) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
    # repeat hash to fill dim
    while len(vals) < dim:
        for b in h:
            vals.append(((b / 255.0) * 2.0) - 1.0)
            if len(vals) >= dim:
                break
    return vals

# ----- Basic health/root -----
@api.get("/")
async def root():
    return {"message": "AI-Reflect API is up"}

@api.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(client_name=input.client_name)
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    await db.status_checks.insert_one(doc)
    return status_obj

@api.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    for check in status_checks:
        if isinstance(check.get('timestamp'), str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    return status_checks

# ----- Project Endpoints -----
@api.post("/projects", response_model=Project)
async def create_project(payload: ProjectCreate):
    proj = Project(**payload.model_dump())
    doc = await serialize_dt(proj.model_dump())
    await db.projects.insert_one({**doc, "_id": proj.id})
    return proj

@api.get("/projects", response_model=List[Project])
async def list_projects():
    docs = await db.projects.find({}, {"_id": 0}).to_list(1000)
    for d in docs:
        for key in ("created_at", "updated_at"):
            if key in d and isinstance(d[key], str):
                d[key] = datetime.fromisoformat(d[key])
    return docs

@api.get("/projects/{project_id}", response_model=Project)
async def get_project(project_id: str):
    doc = await db.projects.find_one({"id": project_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    for key in ("created_at", "updated_at"):
        if key in doc and isinstance(doc[key], str):
            doc[key] = datetime.fromisoformat(doc[key])
    return doc

@api.patch("/projects/{project_id}", response_model=Project)
async def update_project(project_id: str, payload: ProjectUpdate):
    updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    res = await db.projects.update_one({"id": project_id}, {"$set": updates})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")
    doc = await db.projects.find_one({"id": project_id}, {"_id": 0})
    for key in ("created_at", "updated_at"):
        if key in doc and isinstance(doc[key], str):
            doc[key] = datetime.fromisoformat(doc[key])
    return doc

# ----- Reflection Endpoints (with AI + Embeddings) -----
@api.post("/reflections", response_model=Reflection)
async def create_reflection(payload: ReflectionCreate):
    # Ensure project exists
    proj = await db.projects.find_one({"id": payload.project_id})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    text: Optional[str] = payload.text

    if not text:
        if not payload.prompt:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'prompt' to generate reflection")
        # Generate text via OpenAI chat
        client_ai = get_openai_client()
        try:
            completion = await client_ai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a concise research assistant. Create a short self-reflection (2-4 sentences) about the project context for market reasoning."},
                    {"role": "user", "content": payload.prompt},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            text = completion.choices[0].message.content or ""
        except Exception:
            # Fallback to a deterministic reflection if AI is unavailable (e.g., missing key or network)
            text = f"Auto-reflection (fallback): Based on the prompt, focus on reducing overfitting, validating signals against multiple timeframes, and weighting volume/volatility to avoid chasing noise. Prompt: {payload.prompt[:180]}"

    # Generate embedding for text
    vector: List[float] = []
    try:
        client_ai = get_openai_client()
        emb = await client_ai.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=text,
            encoding_format="float",
        )
        vector = emb.data[0].embedding  # type: ignore
    except Exception as e:
        logging.getLogger(__name__).warning(f"Embedding generation failed: {e}")
        vector = fallback_embed(text)

    ref = Reflection(project_id=payload.project_id, text=text, vector=vector)
    doc = await serialize_dt(ref.model_dump())
    await db.reflections.insert_one({**doc, "_id": ref.id})
    return ref

@api.get("/reflections", response_model=List[Reflection])
async def list_reflections(project_id: Optional[str] = Query(default=None)):
    filt = {"project_id": project_id} if project_id else {}
    docs = await db.reflections.find(filt, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for d in docs:
        if isinstance(d.get('created_at'), str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
    return docs

@api.get("/reflections/{ref_id}", response_model=Reflection)
async def get_reflection(ref_id: str):
    doc = await db.reflections.find_one({"id": ref_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="Reflection not found")
    if isinstance(doc.get('created_at'), str):
        doc['created_at'] = datetime.fromisoformat(doc['created_at'])
    return doc

# ----- Anomaly Endpoints -----
@api.post("/anomalies", response_model=Anomaly)
async def create_anomaly(payload: AnomalyCreate):
    proj = await db.projects.find_one({"id": payload.project_id})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    an = Anomaly(**payload.model_dump())
    doc = await serialize_dt(an.model_dump())
    await db.anomalies.insert_one({**doc, "_id": an.id})
    return an

@api.get("/anomalies", response_model=List[Anomaly])
async def list_anomalies(project_id: Optional[str] = Query(default=None)):
    filt = {"project_id": project_id} if project_id else {}
    docs = await db.anomalies.find(filt, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for d in docs:
        if isinstance(d.get('created_at'), str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
    return docs

# ----- Compute: Clustering & Surprise Detection -----
@api.post("/compute/clusters")
async def compute_clusters(k: int = Query(default=4, ge=2, le=12)):
    # lazy import sklearn
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing dependencies for clustering: {e}")

    refs = await db.reflections.find({}, {"_id": 0, "id": 1, "vector": 1}).to_list(10000)
    if not refs:
        return {"assigned": 0}

    X = []
    ids = []
    for r in refs:
        vec = r.get("vector") or []
        if not vec:
            # ensure stable vector for clustering
            vec = fallback_embed(r.get("id", ""))
        X.append(vec)
        ids.append(r["id"])

    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e}")

    # persist cluster on reflection docs
    for rid, lab in zip(ids, labels):
        await db.reflections.update_one({"id": rid}, {"$set": {"cluster": int(lab)}})

    return {"assigned": len(ids), "clusters": int(k)}

@api.post("/compute/anomalies")
async def compute_anomalies(contamination: float = Query(default=0.05, gt=0.0, lt=0.5)):
    try:
        from sklearn.ensemble import IsolationForest  # type: ignore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Missing dependencies for anomalies: {e}")

    refs = await db.reflections.find({}, {"_id": 0, "id": 1, "project_id": 1, "vector": 1}).to_list(10000)
    if not refs:
        return {"flagged": 0}

    X = []
    ids = []
    proj_map = {}
    for r in refs:
        vec = r.get("vector") or []
        if not vec:
            vec = fallback_embed(r.get("id", ""))
        X.append(vec)
        ids.append(r["id"])
        proj_map[r["id"]] = r["project_id"]

    try:
        iso = IsolationForest(contamination=contamination, random_state=42)
        preds = iso.fit_predict(X)  # -1 anomalies, 1 normal
        scores = iso.decision_function(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {e}")

    flagged = 0
    for i, rid in enumerate(ids):
        if preds[i] == -1:
            flagged += 1
            sev = 'high' if scores[i] < -0.2 else 'medium'
            detail = f"Auto anomaly: outlier reflection {rid} (score={scores[i]:.3f})"
            an = Anomaly(project_id=proj_map[rid], detail=detail, severity=sev)
            doc = await serialize_dt(an.model_dump())
            await db.anomalies.insert_one({**doc, "_id": an.id})

    return {"flagged": flagged}

# ----- Graph endpoint (for D3 force-graph) -----
@api.get("/graph")
async def graph_data():
    projects = await db.projects.find({}, {"_id": 0}).to_list(1000)
    reflections = await db.reflections.find({}, {"_id": 0}).to_list(5000)
    anomalies = await db.anomalies.find({}, {"_id": 0}).to_list(5000)

    nodes = []
    links = []

    # Projects
    for p in projects:
        nodes.append({
            "id": p["id"],
            "type": "project",
            "label": p["title"],
            "status": p.get("status", "draft"),
        })

    # Optional cluster nodes
    clusters = {}
    for r in reflections:
        if r.get("cluster") is not None:
            cid = f"cluster-{r['cluster']}"
            clusters[cid] = r['cluster']

    for cid, lab in clusters.items():
        nodes.append({
            "id": cid,
            "type": "cluster",
            "label": f"Cluster {lab}",
        })

    # Reflections
    for r in reflections:
        rid = r["id"]
        nodes.append({
            "id": rid,
            "type": "reflection",
            "label": (r.get("text", "")[:60] + ("…" if len(r.get("text", "")) > 60 else "")),
        })
        links.append({"source": r["project_id"], "target": rid, "kind": "has_reflection"})
        if r.get("cluster") is not None:
            links.append({"source": rid, "target": f"cluster-{r['cluster']}", "kind": "in_cluster", "weight": 1})

    # Anomalies
    for a in anomalies:
        aid = a["id"]
        nodes.append({
            "id": aid,
            "type": "anomaly",
            "label": a.get("severity", "low"),
            "severity": a.get("severity", "low"),
        })
        links.append({"source": a["project_id"], "target": aid, "kind": "has_anomaly", "weight": 1})

    # Simple score: reflections - anomalies (clamped)
    score = max(0, len(reflections) * 2 - len(anomalies))

    return {"nodes": nodes, "links": links, "stats": {"score": score, "projects": len(projects), "reflections": len(reflections), "anomalies": len(anomalies)}}

@api.post("/images/generate", response_model=ImageResponse)
async def generate_image(payload: ImageRequest):
    client_ai = get_openai_client()
    try:
        result = await client_ai.images.generate(
            model="gpt-image-1",
            prompt=payload.prompt,
            size=payload.size,
            quality=payload.quality,
        )
        url = result.data[0].url  # type: ignore[attr-defined]
        return ImageResponse(url=url)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Image generation failed: {e}")
        return ImageResponse(url="https://placehold.co/512x512/png?text=AI+Image+error")

# ----- AI health -----
@api.get("/ai/health", response_model=AIHealth)
async def ai_health():
    if not OPENAI_KEY:
        return AIHealth(ok=False, reason="missing_key")
    try:
        client_ai = get_openai_client()
        # cheap call: 1 token embed
        await client_ai.embeddings.create(model=OPENAI_EMBED_MODEL, input="ok")
        return AIHealth(ok=True)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg or "invalid_api_key" in msg:
            return AIHealth(ok=False, reason="unauthorized")
        return AIHealth(ok=False, reason="error")

# ----- Reflection scheduler (every N hours) -----
SCHEDULE_HOURS = float(os.environ.get('REFLECTION_SCHEDULE_HOURS', '4'))
SCHED_NEXT: Optional[str] = None

async def _auto_reflect_once():
    global SCHED_NEXT
    # choose most recent project or any
    proj = await db.projects.find_one({}, sort=[("created_at", -1)])
    if proj:
        try:
            await create_reflection(ReflectionCreate(project_id=proj["id"], prompt="Periodic 4h reflection on current market regime and adjustments."))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Scheduled reflection failed: {e}")
    # update next time hint
    SCHED_NEXT = (datetime.now(timezone.utc) + timedelta(hours=SCHEDULE_HOURS)).isoformat()

@api.get("/scheduler/status")
async def scheduler_status():
    return {"next_run_at": SCHED_NEXT}

@app.on_event("startup")
async def startup_tasks():
    global SCHED_NEXT
    # try to start apscheduler
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler  # type: ignore
        sched = AsyncIOScheduler()
        sched.start()
        next_time = datetime.now(timezone.utc) + timedelta(seconds=10)
        SCHED_NEXT = next_time.isoformat()
        sched.add_job(_auto_reflect_once, 'interval', hours=SCHEDULE_HOURS, next_run_time=next_time)
        logging.getLogger(__name__).info("Scheduler started for periodic reflections")
    except Exception as e:
        logging.getLogger(__name__).warning(f"APScheduler not available: {e}")

# Include router
app.include_router(api)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
