from fastapi import FastAPI, APIRouter, HTTPException, Query
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
import re
import random

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

app = FastAPI(title="AI-Reflect MVP")
api = APIRouter(prefix="/api")

# ----- Schemas -----
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

ProjectStatus = Literal['draft', 'live', 'archived']
class ProjectCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    status: ProjectStatus = 'draft'

class Project(ProjectCreate):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
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
    flagged_false_ids: List[str] = Field(default_factory=list)
    deception_detected: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

Severity = Literal['low','medium','high','critical']
class AnomalyCreate(BaseModel):
    project_id: str
    detail: str
    severity: Severity = 'low'

class Anomaly(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_id: str
    detail: str
    severity: Severity
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ImageRequest(BaseModel):
    prompt: str
    size: Optional[str] = "512x512"
    quality: Optional[str] = "standard"

class ImageResponse(BaseModel):
    url: str

class AIHealth(BaseModel):
    ok: bool
    reason: Optional[str] = None

class Candle(BaseModel):
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

PredictionHorizon = Literal['1h','4h','8h','24h','3d','2w','1m']
class Prediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    horizon: PredictionHorizon
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    target_ts: int
    direction: int
    confidence: float
    sentiment: Literal['bullish','neutral','bearish']
    reasoning_text: str = ""
    vector: List[float] = Field(default_factory=list)

class OpponentPrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    persona: Literal['honest','bluffer','chaotic']
    horizon: PredictionHorizon = '1h'
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    base_ts: int
    base_price: float
    target_ts: int
    direction: int
    rationale: str = ""
    evaluated: bool = False
    correct: Optional[bool] = None

class Misinfo(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    headline: str
    source: str = "tweet"
    is_deception: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ----- OpenAI -----
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
            raise HTTPException(500, f"OpenAI SDK not installed: {e}")
        if not OPENAI_KEY:
            raise HTTPException(500, "Missing OPENAI_API_KEY in backend environment")
        async_openai_client = AsyncOpenAI(api_key=OPENAI_KEY)
    return async_openai_client

# ----- Utils -----
async def serialize_dt(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    for k, v in list(out.items()):
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    return out

def fallback_embed(text: str, dim: int = 64) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    vals = []
    while len(vals) < dim:
        for b in h:
            vals.append(((b/255.0)*2.0)-1.0)
            if len(vals) >= dim:
                break
    return vals

_json_block_re = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.I)

def parse_json_relaxed(content: str) -> Dict[str, Any]:
    if not content:
        return {}
    m = _json_block_re.search(content)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start = content.find('{'); end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end+1])
        except Exception:
            return {}
    return {}

# ----- Health -----
@api.get('/')
async def root():
    return {"message": "AI-Reflect API is up"}

@api.get('/ai/health', response_model=AIHealth)
async def ai_health():
    if not OPENAI_KEY:
        return AIHealth(ok=False, reason="missing_key")
    try:
        client_ai = get_openai_client()
        await client_ai.embeddings.create(model=OPENAI_EMBED_MODEL, input="ok")
        return AIHealth(ok=True)
    except Exception as e:
        msg = str(e).lower()
        if '401' in msg or 'unauthorized' in msg:
            return AIHealth(ok=False, reason="unauthorized")
        return AIHealth(ok=False, reason="error")

# ----- Projects -----
@api.post('/projects', response_model=Project)
async def create_project(payload: ProjectCreate):
    proj = Project(**payload.model_dump())
    doc = await serialize_dt(proj.model_dump())
    await db.projects.insert_one({**doc, '_id': proj.id})
    return proj

@api.get('/projects', response_model=List[Project])
async def list_projects():
    docs = await db.projects.find({}, {'_id':0}).to_list(1000)
    for d in docs:
        for k in ('created_at','updated_at'):
            if k in d and isinstance(d[k], str):
                d[k] = datetime.fromisoformat(d[k])
    return docs

@api.patch('/projects/{pid}', response_model=Project)
async def patch_project(pid: str, payload: ProjectUpdate):
    updates = {k:v for k,v in payload.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, 'No fields to update')
    updates['updated_at'] = datetime.now(timezone.utc).isoformat()
    res = await db.projects.update_one({'id': pid}, {'$set': updates})
    if res.matched_count == 0:
        raise HTTPException(404, 'Project not found')
    doc = await db.projects.find_one({'id': pid}, {'_id':0})
    for k in ('created_at','updated_at'):
        if k in doc and isinstance(doc[k], str):
            doc[k] = datetime.fromisoformat(doc[k])
    return doc

# ----- Reflections (with deception flags) -----
@api.post('/reflections', response_model=Reflection)
async def create_reflection(payload: ReflectionCreate):
    proj = await db.projects.find_one({'id': payload.project_id})
    if not proj:
        raise HTTPException(404, 'Project not found')
    text = payload.text
    if not text:
        if not payload.prompt:
            raise HTTPException(400, "Provide either 'text' or 'prompt'")
        try:
            client_ai = get_openai_client()
            completion = await client_ai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[
                    {"role":"system","content":"You are a concise research assistant. Create a short self-reflection (2-4 sentences) about the project context for market reasoning."},
                    {"role":"user","content": payload.prompt},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            text = completion.choices[0].message.content or ""
        except Exception:
            text = f"Auto-reflection (fallback): Based on the prompt, focus on reducing overfitting, validating signals against multiple timeframes, and weighting volume/volatility to avoid chasing noise. Prompt: {payload.prompt[:180]}"

    # Judge recent misinfo (3h)
    since = datetime.now(timezone.utc) - timedelta(hours=3)
    recent = await db.misinfo_events.find({"created_at": {"$gte": since.isoformat()}}, {'_id':0}).sort('created_at', -1).to_list(20)
    flagged_ids: List[str] = []
    deception_detected = False
    if OPENAI_KEY and recent:
        try:
            client_ai = get_openai_client()
            info_blob = json.dumps([{k:v for k,v in m.items() if k in ('id','headline','source','created_at')} for m in recent])
            jprompt = "You may be lied to. Return JSON {flagged_ids: [ids_of_false_or_deceptive]} based on these items:\n" + info_blob
            r = await client_ai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role":"system","content":"Judge misinformation succinctly."},{"role":"user","content": jprompt}],
                temperature=0,
                max_tokens=150,
            )
            parsed = parse_json_relaxed(r.choices[0].message.content or '{}')
            if isinstance(parsed, dict):
                flagged_ids = [str(x) for x in (parsed.get('flagged_ids') or [])]
                deception_detected = bool(flagged_ids)
        except Exception as e:
            logging.getLogger(__name__).warning(f"misinfo judge error: {e}")

    # Embedding
    try:
        client_ai = get_openai_client()
        emb = await client_ai.embeddings.create(model=OPENAI_EMBED_MODEL, input=text, encoding_format='float')
        vector = emb.data[0].embedding  # type: ignore
    except Exception as e:
        logging.getLogger(__name__).warning(f"Embedding generation failed: {e}")
        vector = fallback_embed(text)

    ref = Reflection(project_id=payload.project_id, text=text, vector=vector, flagged_false_ids=flagged_ids, deception_detected=deception_detected)
    doc = await serialize_dt(ref.model_dump())
    await db.reflections.insert_one({**doc, '_id': ref.id})

    # Update deception metrics
    await metrics_init()
    if recent:
        # Any flagged not deception -> FP, flagged & deception -> TP etc.
        for mi in recent:
            is_dec = bool(mi.get('is_deception', True))
            flagged = mi['id'] in flagged_ids
            if flagged and is_dec:
                await db.metrics.update_one({'_id':'global'}, {'$inc': {'deception.tp': 1}})
            elif flagged and not is_dec:
                await db.metrics.update_one({'_id':'global'}, {'$inc': {'deception.fp': 1}})
            elif (not flagged) and is_dec:
                await db.metrics.update_one({'_id':'global'}, {'$inc': {'deception.fn': 1}})
            else:
                await db.metrics.update_one({'_id':'global'}, {'$inc': {'deception.tn': 1}})

    return ref

@api.get('/reflections', response_model=List[Reflection])
async def list_reflections(project_id: Optional[str] = None):
    filt = {'project_id': project_id} if project_id else {}
    docs = await db.reflections.find(filt, {'_id':0}).sort('created_at', -1).to_list(1000)
    for d in docs:
        if isinstance(d.get('created_at'), str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
    return docs

# ----- Anomalies -----
@api.post('/anomalies', response_model=Anomaly)
async def create_anomaly(payload: AnomalyCreate):
    if not await db.projects.find_one({'id': payload.project_id}):
        raise HTTPException(404, 'Project not found')
    an = Anomaly(**payload.model_dump())
    doc = await serialize_dt(an.model_dump())
    await db.anomalies.insert_one({**doc, '_id': an.id})
    return an

@api.get('/anomalies', response_model=List[Anomaly])
async def list_anomalies(project_id: Optional[str] = None):
    filt = {'project_id': project_id} if project_id else {}
    docs = await db.anomalies.find(filt, {'_id':0}).sort('created_at', -1).to_list(1000)
    for d in docs:
        if isinstance(d.get('created_at'), str):
            d['created_at'] = datetime.fromisoformat(d['created_at'])
    return docs

# ----- Live data (CoinGecko Demo/Pro + fallback) -----
DATA_SOURCE = os.environ.get('DATA_SOURCE','coingecko')
CG_PRO_KEY = os.environ.get('COINGECKO_API_KEY')
CG_DEMO_KEY = os.environ.get('COINGECKO_DEMO_API_KEY')
_last_candle_ts: Optional[int] = None
_ingest_backoff: int = 60

async def fetch_latest_candle_coingecko() -> bool:
    import httpx
    try:
        base = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
        headers = {}
        params = {'vs_currency':'usd','days':'1','interval':'minute'}
        if CG_PRO_KEY:
            base = 'https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart'
            headers['x-cg-pro-api-key'] = CG_PRO_KEY
        elif CG_DEMO_KEY:
            headers['x-cg-demo-api-key'] = CG_DEMO_KEY
            params['x_cg_demo_api_key'] = CG_DEMO_KEY
        async with httpx.AsyncClient(timeout=20) as hc:
            r = await hc.get(base, params=params, headers=headers)
            if r.status_code >= 400:
                logging.getLogger(__name__).warning(f"coingecko status={r.status_code} text={r.text[:160]}")
            r.raise_for_status()
            data = r.json()
        prices = data.get('prices',[]); vols = data.get('total_volumes',[])
        if not prices:
            return False
        ts_ms, price = prices[-1]; vol = vols[-1][1] if vols else 0.0
        candle = {'ts': int(ts_ms//1000),'open': float(price),'high': float(price),'low': float(price),'close': float(price),'volume': float(vol)}
        await db.price_candles.update_one({'ts': candle['ts']}, {'$set': candle}, upsert=True)
        globals()['_last_candle_ts'] = candle['ts']
        return True
    except Exception as e:
        logging.getLogger(__name__).warning(f"ingest coingecko error: {e}")
        # fallback to simple price
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as hc:
                sp = await hc.get('https://api.coingecko.com/api/v3/simple/price', params={'ids':'bitcoin','vs_currencies':'usd', **({'x_cg_demo_api_key': CG_DEMO_KEY} if CG_DEMO_KEY else {})}, headers=({'x-cg-demo-api-key': CG_DEMO_KEY} if CG_DEMO_KEY else {}))
                sp.raise_for_status(); j = sp.json(); price = float(j.get('bitcoin',{}).get('usd') or 0)
                if price>0:
                    ts = int(datetime.now(timezone.utc).timestamp())
                    candle = {'ts': ts,'open': price,'high': price,'low': price,'close': price,'volume': 0.0}
                    await db.price_candles.update_one({'ts': ts}, {'$set': candle}, upsert=True)
                    globals()['_last_candle_ts'] = ts
                    return True
        except Exception as e2:
            logging.getLogger(__name__).warning(f"simple price fallback failed: {e2}")
        return False

async def fetch_latest_candle() -> bool:
    return await fetch_latest_candle_coingecko()

@api.get('/data/health')
async def data_health():
    now = int(datetime.now(timezone.utc).timestamp())
    age = (now - _last_candle_ts) if _last_candle_ts else None
    ok = age is not None and age < 180
    return {'source': DATA_SOURCE, 'ok': bool(ok), 'last_ts': _last_candle_ts, 'age_sec': age}

@api.post('/data/ingest_once')
async def ingest_once():
    ok = await fetch_latest_candle()
    return {'ok': ok}

@api.get('/candles/latest', response_model=Candle)
async def candles_latest():
    doc = await db.price_candles.find_one({}, sort=[('ts',-1)], projection={'_id':0})
    if not doc:
        raise HTTPException(404, 'No candles yet')
    return doc

@api.get('/ticker')
async def ticker():
    latest = await db.price_candles.find_one({}, sort=[('ts',-1)], projection={'_id':0})
    if not latest:
        raise HTTPException(404, 'No candles yet')
    target_ts = latest['ts'] - 86400
    past = await db.price_candles.find_one({'ts': {'$lte': target_ts}}, sort=[('ts',-1)], projection={'_id':0})
    change_pct = 0.0
    if past and past.get('close'):
        change_pct = ((latest['close'] - past['close']) / past['close']) * 100.0
    return {'price': latest['close'], 'change24h': change_pct}

# ----- Predictions (LLM) -----
HORIZONS: List[tuple[str,int]] = [('1h',60),('4h',240),('8h',480),('24h',1440),('3d',4320),('2w',20160),('1m',43200)]

@api.post('/predictions/run')
async def predictions_run():
    return await predict_once()

@api.get('/predictions/latest')
async def predictions_latest():
    out = []
    for h,_ in HORIZONS:
        doc = await db.predictions.find_one({'horizon': h}, sort=[('created_at',-1)], projection={'_id':0})
        if doc:
            out.append(doc)
    return out

@api.get('/predictions/history')
async def predictions_history(horizon: PredictionHorizon, limit: int = Query(50, ge=1, le=500)):
    docs = await db.predictions.find({'horizon': horizon}, {'_id':0}).sort('created_at',-1).limit(limit).to_list(length=limit)
    return docs

async def predict_once() -> Dict[str, Any]:
    cutoff = int(datetime.now(timezone.utc).timestamp()) - 60*60
    candles = await db.price_candles.find({'ts': {'$gte': cutoff}}, {'_id':0}).sort('ts',1).to_list(200)
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    rows = "\n".join(f"{c['ts']}: {c['close']:.2f}" for c in candles[-60:])
    prompt = "Given the last 60 one-minute BTC closes, output JSON per horizon ['1h','4h','8h','24h','3d','2w','1m'] with keys dir(0/1/2), conf(0..1), sent('bullish'|'neutral'|'bearish'), explain(<=200 chars).\n" + rows
    outputs: Dict[str, Dict[str, Any]] = {}
    used_fallback = False
    if OPENAI_KEY:
        try:
            client_ai = get_openai_client()
            resp = await client_ai.chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                messages=[{"role":"system","content":"You are a BTC analyst."},{"role":"user","content": prompt}],
                temperature=0.3,
                max_tokens=400,
            )
            parsed = parse_json_relaxed(resp.choices[0].message.content or '{}')
            if isinstance(parsed, dict) and parsed:
                outputs = parsed
        except Exception as e:
            logging.getLogger(__name__).warning(f"prediction openai error: {e}")
            used_fallback = True
    else:
        used_fallback = True
    if used_fallback or not outputs:
        outputs = {h:{'dir':1,'conf':0.0,'sent':'neutral','explain':'fallback'} for h,_ in HORIZONS}
    inserted = []
    for h, mins in HORIZONS:
        item = outputs.get(h) or {'dir':1,'conf':0.0,'sent':'neutral','explain':'fallback'}
        dir_i = int(item.get('dir',1)); conf = float(item.get('conf',0.0)); sent = str(item.get('sent','neutral')); explain = str(item.get('explain',''))
        try:
            client_ai = get_openai_client()
            emb = await client_ai.embeddings.create(model=OPENAI_EMBED_MODEL, input=explain or 'prediction', encoding_format='float')
            vec = emb.data[0].embedding  # type: ignore
        except Exception as e:
            logging.getLogger(__name__).warning(f"prediction embed error: {e}")
            vec = fallback_embed(explain or h)
        doc = Prediction(horizon=h, target_ts=int(now.timestamp())+mins*60, direction=dir_i, confidence=conf, sentiment=(sent if sent in ('bullish','neutral','bearish') else 'neutral'), reasoning_text=explain, vector=vec)
        dumped = await serialize_dt(doc.model_dump()); await db.predictions.insert_one({**dumped, '_id': doc.id}); inserted.append(doc.id)
    return {'inserted': inserted}

# ----- Opponent simulator -----
@api.post('/opponent/predictions')
async def opponent_predict(persona: Literal['honest','bluffer','chaotic'] = 'honest', horizon: PredictionHorizon = '1h'):
    cutoff = int(datetime.now(timezone.utc).timestamp()) - 60*60
    candles = await db.price_candles.find({'ts': {'$gte': cutoff}}, {'_id':0}).sort('ts',1).to_list(200)
    if not candles:
        raise HTTPException(400, 'No candles to compute opponent prediction')
    closes = [c['close'] for c in candles[-60:]]
    base = candles[-1]
    def momentum(seq):
        if len(seq)<2: return 'same'
        return 'higher' if seq[-1]>seq[0] else ('lower' if seq[-1]<seq[0] else 'same')
    if persona=='honest': d = momentum(closes); rationale = 'Momentum-based call from last 60m trend'
    elif persona=='bluffer': d = random.choice(['higher','lower']); rationale = 'Confident call without evidence'
    else: d = random.choice(['higher','lower','same']); rationale = 'Chaotic random direction'
    dir_map = {'lower':0,'same':1,'higher':2}
    mins = dict(HORIZONS)[horizon]
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    op = OpponentPrediction(persona=persona, horizon=horizon, base_ts=base['ts'], base_price=float(base['close']), target_ts=int(now.timestamp())+mins*60, direction=dir_map[d], rationale=rationale)
    dumped = await serialize_dt(op.model_dump()); await db.opponent_predictions.insert_one({**dumped, '_id': op.id})
    return op

# ----- Misinfo injection -----
MISINFO_TEMPLATES = [
    'BREAKING: BTC supply doubled overnight due to chain bug',
    'Leaked memo: Major exchange halting BTC withdrawals indefinitely',
    'Analyst claims BTC fair value is $0 per IMF report',
]

@api.post('/misinfo/inject', response_model=Misinfo)
async def misinfo_inject(headline: Optional[str] = None, source: str = 'tweet', is_deception: bool = True):
    hl = headline or random.choice(MISINFO_TEMPLATES)
    mi = Misinfo(headline=hl, source=source, is_deception=is_deception)
    dumped = await serialize_dt(mi.model_dump()); await db.misinfo_events.insert_one({**dumped, '_id': mi.id})
    return mi

@api.get('/misinfo/latest')
async def misinfo_latest(limit: int = Query(5, ge=1, le=20)):
    return await db.misinfo_events.find({}, {'_id':0}).sort('created_at', -1).limit(limit).to_list(length=limit)

# ----- Metrics -----
async def metrics_init():
    await db.metrics.update_one({'_id':'global'}, {'$setOnInsert': {'opponent': {'honest': {'correct':0,'total':0}, 'bluffer': {'correct':0,'total':0}, 'chaotic': {'correct':0,'total':0}}, 'deception': {'tp':0,'fp':0,'fn':0,'tn':0}}}, upsert=True)

@api.get('/metrics/trust')
async def metrics_trust():
    await metrics_init()
    m = await db.metrics.find_one({'_id':'global'}, {'_id':0,'opponent':1})
    out = {}
    for persona, vals in (m.get('opponent') or {}).items():
        correct = int(vals.get('correct',0)); total = int(vals.get('total',0)); acc = (correct/total) if total>0 else 0.0
        out[persona] = {'accuracy': acc, 'correct': correct, 'total': total}
    return out

@api.get('/metrics/deception')
async def metrics_deception():
    await metrics_init()
    m = await db.metrics.find_one({'_id':'global'}, {'_id':0,'deception':1})
    d = m.get('deception') or {'tp':0,'fp':0,'fn':0,'tn':0}
    tp,fp,fn,tn = int(d.get('tp',0)), int(d.get('fp',0)), int(d.get('fn',0)), int(d.get('tn',0))
    precision = (tp/(tp+fp)) if (tp+fp)>0 else 0.0
    recall = (tp/(tp+fn)) if (tp+fn)>0 else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall}

async def evaluate_opponents_once():
    await metrics_init()
    now = int(datetime.now(timezone.utc).timestamp())
    cursor = db.opponent_predictions.find({'evaluated': False, 'target_ts': {'$lte': now}}, {'_id':0})
    async for op in cursor:
        # Find candle at/after target_ts
        target = await db.price_candles.find_one({'ts': {'$gte': op['target_ts']}}, sort=[('ts',1)], projection={'_id':0})
        if not target:
            continue
        base_price = float(op['base_price']); actual = float(target['close'])
        eps = 0.001  # 0.1%
        if actual > base_price*(1+eps): actual_dir = 2
        elif actual < base_price*(1-eps): actual_dir = 0
        else: actual_dir = 1
        correct = (actual_dir == int(op['direction']))
        # mark evaluated
        await db.opponent_predictions.update_one({'id': op['id']}, {'$set': {'evaluated': True, 'correct': bool(correct)}})
        # increment metrics
        path_total = f"opponent.{op['persona']}.total"; path_correct = f"opponent.{op['persona']}.correct"
        await db.metrics.update_one({'_id':'global'}, {'$inc': {path_total: 1, path_correct: (1 if correct else 0)}})

# ----- Graph -----
@api.get('/graph')
async def graph_data():
    projects = await db.projects.find({}, {'_id':0}).to_list(1000)
    reflections = await db.reflections.find({}, {'_id':0}).to_list(5000)
    anomalies = await db.anomalies.find({}, {'_id':0}).to_list(5000)
    latest_preds = await predictions_latest()
    misinfos = await db.misinfo_events.find({}, {'_id':0}).sort('created_at', -1).limit(20).to_list(length=20)

    nodes = []; links = []
    for p in projects:
        nodes.append({'id': p['id'],'type':'project','label': p['title'],'status': p.get('status','draft')})
    clusters = {}
    for r in reflections:
        if r.get('cluster') is not None:
            cid = f"cluster-{r['cluster']}"; clusters[cid] = r['cluster']
    for cid, lab in clusters.items():
        nodes.append({'id': cid, 'type':'cluster','label': f'Cluster {lab}'})
    for r in reflections:
        rid = r['id']
        nodes.append({'id': rid,'type':'reflection','label': (r.get('text','')[:60] + ('…' if len(r.get('text',''))>60 else ''))})
        links.append({'source': r['project_id'],'target': rid,'kind':'has_reflection','weight':1})
        if r.get('cluster') is not None:
            links.append({'source': rid,'target': f"cluster-{r['cluster']}",'kind':'in_cluster','weight':1})
    for a in anomalies:
        aid = a['id']; nodes.append({'id': aid,'type':'anomaly','label': a.get('severity','low'),'severity': a.get('severity','low')}); links.append({'source': a['project_id'],'target': aid,'kind':'has_anomaly','weight':1})
    icon = {0:'↓',1:'→',2:'↑'}
    for pdoc in latest_preds:
        pid = f"pred-{pdoc['horizon']}-{pdoc.get('target_ts',0)}"; label = f"{icon.get(int(pdoc.get('direction',1)),'→')} {pdoc['horizon']}"
        nodes.append({'id': pid,'type':'prediction','label': label,'sentiment': pdoc.get('sentiment','neutral'),'confidence': pdoc.get('confidence',0.0)})
        proj = await db.projects.find_one({}, sort=[('updated_at',-1)], projection={'_id':0,'id':1})
        if proj:
            links.append({'source': proj['id'],'target': pid,'kind':'has_prediction','weight':1})
    for m in misinfos:
        nodes.append({'id': m['id'],'type':'misinfo','label': m.get('headline','misinfo'),'is_deception': m.get('is_deception', True)})
        for r in reflections:
            try:
                rc = r.get('created_at'); mc = m.get('created_at')
                if isinstance(rc,str): rc = datetime.fromisoformat(rc)
                if isinstance(mc,str): mc = datetime.fromisoformat(mc)
                if abs((rc - mc).total_seconds()) <= 3*3600:
                    flagged = m['id'] in (r.get('flagged_false_ids') or [])
                    links.append({'source': m['id'],'target': r['id'],'kind':'misinfo_context','weight':1,'flagged': flagged})
            except Exception:
                continue
    score = max(0, len(reflections)*2 - len(anomalies))
    return {'nodes': nodes,'links': links,'stats': {'score': score,'projects': len(projects),'reflections': len(reflections),'anomalies': len(anomalies)}}

# ----- Schedulers -----
SCHEDULE_HOURS = float(os.environ.get('REFLECTION_SCHEDULE_HOURS','4'))
SCHED_NEXT: Optional[str] = None
PRED_NEXT: Optional[str] = None
MISINFO_NEXT: Optional[str] = None

async def _auto_reflect_once():
    global SCHED_NEXT
    proj = await db.projects.find_one({}, sort=[('created_at',-1)])
    if proj:
        try:
            await create_reflection(ReflectionCreate(project_id=proj['id'], prompt='Periodic 4h reflection on current market regime and adjustments.'))
        except Exception as e:
            logging.getLogger(__name__).warning(f"Scheduled reflection failed: {e}")
    SCHED_NEXT = (datetime.now(timezone.utc) + timedelta(hours=SCHEDULE_HOURS)).isoformat()

async def _tick_ingest():
    global _ingest_backoff
    while True:
        ok = await fetch_latest_candle()
        _ingest_backoff = 60 if ok else min(300, _ingest_backoff+30)
        await asyncio.sleep(_ingest_backoff)

async def _schedule_predictions():
    global PRED_NEXT
    while True:
        now = datetime.now(timezone.utc); next_top = (now + timedelta(hours=1)).replace(minute=0, second=5, microsecond=0); PRED_NEXT = next_top.isoformat()
        await asyncio.sleep((next_top - now).total_seconds())
        try:
            await predict_once()
        except Exception as e:
            logging.getLogger(__name__).warning(f"predict loop error: {e}")

async def _schedule_misinfo():
    global MISINFO_NEXT
    while True:
        now = datetime.now(timezone.utc); next_time = now + timedelta(hours=2); MISINFO_NEXT = next_time.isoformat()
        await asyncio.sleep((next_time - now).total_seconds())
        try:
            await misinfo_inject()
        except Exception as e:
            logging.getLogger(__name__).warning(f"misinfo inject error: {e}")

async def _schedule_evaluator():
    while True:
        try:
            await evaluate_opponents_once()
        except Exception as e:
            logging.getLogger(__name__).warning(f"opponent evaluator error: {e}")
        await asyncio.sleep(300)

@api.get('/scheduler/status')
async def scheduler_status():
    return {'next_reflection': SCHED_NEXT, 'next_prediction': PRED_NEXT, 'next_misinfo': MISINFO_NEXT}

@app.on_event('startup')
async def startup_tasks():
    try:
        await metrics_init()
        await db.price_candles.create_index('ts', unique=True)
        await db.predictions.create_index([('horizon',1),('created_at',-1)])
        await db.opponent_predictions.create_index([('persona',1),('created_at',-1)])
        await db.misinfo_events.create_index('created_at')
    except Exception as e:
        logging.getLogger(__name__).warning(f"index create error: {e}")
    try:
        asyncio.create_task(_tick_ingest())
        asyncio.create_task(_schedule_predictions())
        asyncio.create_task(_schedule_misinfo())
        asyncio.create_task(_schedule_evaluator())
        global SCHED_NEXT
        SCHED_NEXT = (datetime.now(timezone.utc) + timedelta(seconds=10)).isoformat()
    except Exception as e:
        logging.getLogger(__name__).warning(f"background loops error: {e}")

app.include_router(api)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS','*').split(','),
    allow_methods=['*'],
    allow_headers=['*'],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.on_event('shutdown')
async def shutdown_db_client():
    client.close()
