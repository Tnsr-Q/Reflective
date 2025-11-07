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
import asyncio
import json
import hashlib

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

# Price candles
class Candle(BaseModel):
    ts: int  # epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float

# Predictions
PredictionHorizon = Literal['1h','4h','8h','24h','3d','2w','1m']
class Prediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    horizon: PredictionHorizon
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_ts: int
    direction: int = Field(ge=0, le=2)  # 0 lower, 1 same, 2 higher
    confidence: float = Field(ge=0, le=1)
    sentiment: Literal['bullish','neutral','bearish']
    reasoning_text: str = ""
    vector: List[float] = Field(default_factory=list)

# ----- OpenAI (via env) -----
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_CHAT_MODEL = os.environ.get('OPENAI_CHAT_MODEL', 'gpt-4o-mini')
OPENAI_EMBED_MODEL = os.environ.get('OPENAI_EMBED_MODEL', 'text-embedding-3-small')

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

def fallback_embed(text: str, dim: int = 64) -> List[float]:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vals = []
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

# ----- Reflection Endpoints (AI + Embeddings) -----
@api.post("/reflections", response_model=Reflection)
async def create_reflection(payload: ReflectionCreate):
    proj = await db.projects.find_one({"id": payload.project_id})
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")

    text: Optional[str] = payload.text

    if not text:
        if not payload.prompt:
            raise HTTPException(status_code=400, detail="Provide either 'text' or 'prompt' to generate reflection")
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
            text = f"Auto-reflection (fallback): Based on the prompt, focus on reducing overfitting, validating signals against multiple timeframes, and weighting volume/volatility to avoid chasing noise. Prompt: {payload.prompt[:180]}"

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
            vec = fallback_embed(r.get("id", ""))
        X.append(vec)
        ids.append(r["id"])

    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {e}")

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
        await client_ai.embeddings.create(model=OPENAI_EMBED_MODEL, input="ok")
        return AIHealth(ok=True)
    except Exception as e:
        msg = str(e).lower()
        if "401" in msg or "unauthorized" in msg or "invalid_api_key" in msg:
            return AIHealth(ok=False, reason="unauthorized")
        return AIHealth(ok=False, reason="error")

# ----- Live data ingestion (CoinGecko free/pro) -----
DATA_SOURCE = os.environ.get('DATA_SOURCE', 'coingecko')
CG_KEY = os.environ.get('COINGECKO_API_KEY')
_last_candle_ts: Optional[int] = None

async def fetch_latest_candle():
    import httpx
    global _last_candle_ts
    try:
        if DATA_SOURCE == 'coingecko':
            if CG_KEY:
                base = "https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart"
                headers = {"x-cg-pro-api-key": CG_KEY}
            else:
                base = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
                headers = {}
            params = {"vs_currency": "usd", "days": "1", "interval": "minutely"}
            async with httpx.AsyncClient(timeout=20) as hc:
                r = await hc.get(base, params=params, headers=headers)
                r.raise_for_status()
                data = r.json()
            prices = data.get("prices", [])
            vols = data.get("total_volumes", [])
            if not prices:
                return
            ts_ms, price = prices[-1]
            vol = vols[-1][1] if vols else 0.0
            candle = {
                "ts": int(ts_ms // 1000),
                "open": float(price),
                "high": float(price),
                "low": float(price),
                "close": float(price),
                "volume": float(vol),
            }
            await db.price_candles.update_one({"ts": candle["ts"]}, {"$set": candle}, upsert=True)
            _last_candle_ts = candle["ts"]
        else:
            # Extend for other sources later
            pass
    except Exception as e:
        logging.getLogger(__name__).warning(f"ingest error: {e}")

@api.get("/data/health")
async def data_health():
    now = int(datetime.now(timezone.utc).timestamp())
    age = (now - _last_candle_ts) if _last_candle_ts else None
    ok = age is not None and age < 180
    return {"source": DATA_SOURCE, "ok": bool(ok), "last_ts": _last_candle_ts, "age_sec": age}

@api.get("/candles/latest", response_model=Candle)
async def candles_latest():
    doc = await db.price_candles.find_one({}, sort=[("ts", -1)], projection={"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="No candles yet")
    return doc

@api.get("/ticker")
async def ticker():
    latest = await db.price_candles.find_one({}, sort=[("ts", -1)], projection={"_id": 0})
    if not latest:
        raise HTTPException(status_code=404, detail="No candles yet")
    # compute 24h change if we have old candle
    target_ts = latest["ts"] - 86400
    past = await db.price_candles.find_one({"ts": {"$lte": target_ts}}, sort=[("ts", -1)], projection={"_id": 0})
    change_pct = 0.0
    if past and past.get("close"):
        change_pct = ((latest["close"] - past["close"]) / past["close"]) * 100.0
    return {"price": latest["close"], "change24h": change_pct}

# ----- Prediction Loop -----
HORIZONS: List[tuple[str, int]] = [("1h", 60), ("4h", 240), ("8h", 480), ("24h", 1440), ("3d", 4320), ("2w", 20160), ("1m", 43200)]

async def predict_once() -> Dict[str, Any]:
    # Pull last 60 min candles
    cutoff = int(datetime.now(timezone.utc).timestamp()) - 60 * 60
    candles = await db.price_candles.find({"ts": {"$gte": cutoff}}, {"_id": 0}).sort("ts", 1).to_list(200)
    closes = [c["close"] for c in candles]
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)

    # Build prompt
    text_rows = "\n".join(f"{c['ts']}: {c['close']:.2f}" for c in candles[-60:])
    prompt = (
        "Given the last 60 one-minute BTC closes, output compact JSON with one entry per horizon: "
        "['1h','4h','8h','24h','3d','2w','1m']. Each entry has keys: dir (0 lower/1 same/2 higher), "
        "conf (0..1), sent ('bullish'|'neutral'|'bearish'), explain (<=200 chars).\nData:\n" + text_rows
    )

    outputs: Dict[str, Dict[str, Any]] = {}
    used_fallback = False
    if OPENAI_KEY:
        try:
            client_ai = get_openai_client()
            resp = await client_ai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role": "system", "content": "You are a BTC analyst."}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            content = resp.choices[0].message.content or "{}"
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                outputs = parsed
        except Exception as e:
            logging.getLogger(__name__).warning(f"prediction openai error: {e}")
            used_fallback = True
    else:
        used_fallback = True

    # Fallback: all horizons 'same', conf 0, neutral
    if used_fallback or not outputs:
        outputs = {h: {"dir": 1, "conf": 0.0, "sent": "neutral", "explain": "fallback"} for (h, _) in HORIZONS}

    # Persist predictions
    inserted = []
    for h, mins in HORIZONS:
        item = outputs.get(h) or {"dir": 1, "conf": 0.0, "sent": "neutral", "explain": "fallback"}
        dir_i = int(item.get("dir", 1))
        conf = float(item.get("conf", 0.0))
        sent = str(item.get("sent", "neutral"))
        explain = str(item.get("explain", ""))

        vec: List[float] = []
        try:
            if OPENAI_KEY:
                client_ai = get_openai_client()
                emb = await client_ai.embeddings.create(model=OPENAI_EMBED_MODEL, input=explain or "prediction", encoding_format="float")
                vec = emb.data[0].embedding  # type: ignore
        except Exception as e:
            logging.getLogger(__name__).warning(f"prediction embed error: {e}")
            vec = fallback_embed(explain or h)

        doc = Prediction(
            horizon=h, target_ts=int(now.timestamp()) + mins * 60, direction=dir_i, confidence=conf,
            sentiment=(sent if sent in ("bullish","neutral","bearish") else "neutral"), reasoning_text=explain, vector=vec
        )
        dumped = await serialize_dt(doc.model_dump())
        await db.predictions.insert_one({**dumped, "_id": doc.id})
        inserted.append(doc.id)

    return {"inserted": inserted}

@api.get("/predictions/latest")
async def predictions_latest():
    out = []
    for h, _ in HORIZONS:
        doc = await db.predictions.find_one({"horizon": h}, sort=[("created_at", -1)], projection={"_id": 0})
        if doc:
            out.append(doc)
    return out

@api.get("/predictions/history")
async def predictions_history(horizon: PredictionHorizon, limit: int = Query(default=50, ge=1, le=500)):
    docs = await db.predictions.find({"horizon": horizon}, {"_id": 0}).sort("created_at", -1).limit(limit).to_list(length=limit)
    return docs

@api.post("/predictions/run")
async def predictions_run():
    result = await predict_once()
    return result

# ----- Reflection scheduler + ingest + predictions scheduling -----
SCHEDULE_HOURS = float(os.environ.get('REFLECTION_SCHEDULE_HOURS', '4'))
SCHED_NEXT: Optional[str] = None
PRED_NEXT: Optional[str] = None

async def _auto_reflect_once():
    global SCHED_NEXT
    proj = await db.projects.find_one({}, sort=[("created_at", -1)])
    if proj:
        try:
            await create_reflection(ReflectionCreate(project_id=proj["id"], prompt="Periodic 4h reflection on current market regime and adjustments."))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Scheduled reflection failed: {e}")
    SCHED_NEXT = (datetime.now(timezone.utc) + timedelta(hours=SCHEDULE_HOURS)).isoformat()

async def _tick_ingest():
    while True:
        await fetch_latest_candle()
        await asyncio.sleep(60)

async def _schedule_predictions():
    global PRED_NEXT
    while True:
        # run on top of hour
        now = datetime.now(timezone.utc)
        next_top = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0)
        PRED_NEXT = next_top.isoformat()
        await asyncio.sleep((next_top - now).total_seconds())
        try:
            await predict_once()
        except Exception as e:
            logging.getLogger(__name__).warning(f"predict loop error: {e}")

@api.get("/scheduler/status")
async def scheduler_status():
    return {"next_reflection": SCHED_NEXT, "next_prediction": PRED_NEXT}

@app.on_event("startup")
async def startup_tasks():
    # indexes
    try:
        await db.price_candles.create_index("ts", unique=True)
        await db.predictions.create_index([("horizon", 1), ("created_at", -1)])
    except Exception as e:
        logging.getLogger(__name__).warning(f"index create error: {e}")

    # start background loops (ingest + prediction + reflection via naive loops)
    try:
        asyncio.create_task(_tick_ingest())
        asyncio.create_task(_schedule_predictions())
        # reflection next time seed
        global SCHED_NEXT
        SCHED_NEXT = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
    except Exception as e:
        logging.getLogger(__name__).warning(f"background loops error: {e}")

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
