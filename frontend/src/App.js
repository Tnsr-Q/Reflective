import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import Graph from "./components/Graph";
import PredictionCards from "./components/PredictionCards";
import Ticker from "./components/Ticker";
import TrustMeter from "./components/TrustMeter";
import MisinfoToast from "./components/MisinfoToast";
import {
  createProject,
  listProjects,
  createReflection,
  listReflections,
  createAnomaly,
  listAnomalies,
  fetchGraph,
  computeClusters,
  computeAnomalies,
  getTicker,
  getSchedulerStatus,
  getLatestPredictions,
  fetchMisinfoLatest,
  fetchTrust,
  fetchDeceptionMetrics,
} from "./api/index";

function useGraphData(refreshTick) {
  const [data, setData] = useState({ nodes: [], links: [], stats: { score: 0 } });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const g = await fetchGraph();
        if (mounted) setData(g);
      } catch (e) {
        setError(e?.message || "Failed to load graph");
      } finally {
        setLoading(false);
      }
    };
    run();
    return () => {
      mounted = false;
    };
  }, [refreshTick]);

  return { data, loading, error };
}

function App() {
  const [refreshTick, setRefreshTick] = useState(0);
  const { data, loading, error } = useGraphData(refreshTick);

  const [projects, setProjects] = useState([]);
  const [form, setForm] = useState({ title: "", description: "" });
  const [selectedProjectId, setSelectedProjectId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [creating, setCreating] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [anomDetail, setAnomDetail] = useState("");
  const [anomSeverity, setAnomSeverity] = useState("low");
  const [creatingAnom, setCreatingAnom] = useState(false);

  const [clustersK, setClustersK] = useState(4);
  const [contamination, setContamination] = useState(0.05);
  const [computeBusy, setComputeBusy] = useState(false);

  const [aiBanner, setAIBanner] = useState(null);
  const [dataBanner, setDataBanner] = useState(null);
  const [toast, setToast] = useState(null);

  useEffect(() => {
    let active = true;
    const checkAI = async () => {
      try {
        const res = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/ai/health`);
        const j = await res.json();
        if (!active) return;
        if (!j.ok) {
          const reason = j.reason || "error";
          let msg = "AI is unavailable";
          if (reason === "missing_key") msg = "AI key missing. Add OPENAI_API_KEY in backend/.env and restart backend.";
          if (reason === "unauthorized") msg = "AI unauthorized. Provide a valid OPENAI_API_KEY in backend/.env and restart backend.";
          setAIBanner(msg);
        } else {
          setAIBanner(null);
        }
      } catch (e) {
        if (!active) return;
        setAIBanner("AI status check failed. Try again later.");
      }
    };
    const checkData = async () => {
      try {
        const res2 = await fetch(`${process.env.REACT_APP_BACKEND_URL}/api/data/health`);
        const d = await res2.json();
        if (!active) return;
        if (!d.ok) {
          setDataBanner("Price feed warming up or rate-limited. Using CoinGecko Demo key; will retry.");
        } else {
          setDataBanner(null);
        }
      } catch (e) {
        if (!active) return;
        setDataBanner("Price feed status unavailable.");
      }
    };
    const checkMisinfo = async () => {
      try {
        const items = await fetchMisinfoLatest(1);
        if (!active) return;
        if (Array.isArray(items) && items.length) setToast(items[0]);
      } catch {}
    };
    checkAI(); checkData(); checkMisinfo();
    const id = setInterval(() => { checkAI(); checkData(); checkMisinfo(); }, 30000);
    return () => { active = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    (async () => {
      const list = await listProjects();
      setProjects(list);
      if (!selectedProjectId && list.length) setSelectedProjectId(list[0].id);
    })();

  // Ticker & predictions state
  const [ticker, setTicker] = useState({ price: null, change24h: null });
  const [sched, setSched] = useState({ next_reflection: null, next_prediction: null });
  const [latestPreds, setLatestPreds] = useState([]);

  // Load ticker/scheduler/predictions periodically
  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const [t, s, p] = await Promise.all([
          getTicker().catch(() => ({})),
          getSchedulerStatus().catch(() => ({})),
          getLatestPredictions().catch(() => ([])),
        ]);
        if (!active) return;
        setTicker({ price: t?.price ?? null, change24h: t?.change24h ?? null });
        setSched({ next_reflection: s?.next_reflection ?? null, next_prediction: s?.next_prediction ?? null });
        setLatestPreds(Array.isArray(p) ? p : []);
      } catch (e) {
        // ignore
      }
    };
    load();
    const id = setInterval(load, 15000);
    return () => { active = false; clearInterval(id); };
  }, []);

  }, [refreshTick]);

  const stats = useMemo(() => data?.stats || { score: 0 }, [data]);

  const handleCreateProject = async () => {
    if (!form.title.trim()) return;
    setCreating(true);
    try {
      await createProject({ title: form.title, description: form.description, status: "draft" });
      setForm({ title: "", description: "" });
      setRefreshTick((x) => x + 1);
    } catch (e) { alert(e?.response?.data?.detail || e.message || "Failed to create project"); } finally { setCreating(false); }
  };

  const handleGenerateReflection = async () => {
    if (!selectedProjectId || !prompt.trim()) return;
    setGenerating(true);
    try {
      await createReflection({ project_id: selectedProjectId, prompt });
      setPrompt(""); setRefreshTick((x) => x + 1);
    } catch (e) { alert(e?.response?.data?.detail || e.message || "Failed to generate reflection"); } finally { setGenerating(false); }
  };

  const handleCreateAnomaly = async () => {
    if (!selectedProjectId || !anomDetail.trim()) return;
    setCreatingAnom(true);
    try {
      await createAnomaly({ project_id: selectedProjectId, detail: anomDetail, severity: anomSeverity });
      setAnomDetail(""); setAnomSeverity("low"); setRefreshTick((x) => x + 1);
    } catch (e) { alert(e?.response?.data?.detail || e.message || "Failed to create anomaly"); } finally { setCreatingAnom(false); }
  };

  const handleComputeClusters = async () => {
    setComputeBusy(true);
    try { await computeClusters(clustersK); setRefreshTick((x) => x + 1); }
    catch (e) { alert(e?.response?.data?.detail || e.message || "Cluster compute failed"); }
    finally { setComputeBusy(false); }
  };

  const handleComputeAnomalies = async () => {
    setComputeBusy(true);
    try { await computeAnomalies(contamination); setRefreshTick((x) => x + 1); }
    catch (e) { alert(e?.response?.data?.detail || e.message || "Anomaly detection failed"); }
    finally { setComputeBusy(false); }
  };

  // Metrics
  const [trust, setTrust] = useState(null);
  const [deception, setDeception] = useState(null);
  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const [t, d] = await Promise.all([fetchTrust().catch(() => ({})), fetchDeceptionMetrics().catch(() => ({}))]);
        if (!active) return;
        setTrust(t || {}); setDeception(d || {});
      } catch {}
    };
    load();
    const id = setInterval(load, 20000);
    return () => { active = false; clearInterval(id); };
  }, []);

  return (
    <div className="App">
      <MisinfoToast item={toast} onClose={() => setToast(null)} />
      <header className="header">
        <h1 data-testid="app-title">AI‑Reflect Dashboard</h1>
        <p>Projects, Reflections (AI), Clusters & Anomalies mapped as a force‑graph</p>
      </header>

      {aiBanner && (
        <div data-testid="ai-banner" className="panel" style={{ maxWidth: 1200, margin: "0 auto 12px", borderColor: "#ef4444" }}>
          <strong>Notice:</strong> {aiBanner}
        </div>
      )}
      {dataBanner && (
        <div data-testid="data-banner" className="panel" style={{ maxWidth: 1200, margin: "0 auto 12px", borderColor: "#eab308" }}>
          <strong>Feed:</strong> {dataBanner}
        </div>
      )}

      <main className="container" style={{ maxWidth: 1200, margin: "0 auto", padding: 16 }}>
        <Ticker price={ticker.price} change24h={ticker.change24h} nextPredictionAt={sched.next_prediction} />
        <div style={{ height: 12 }} />
        <PredictionCards items={latestPreds} />
        <div style={{ height: 16 }} />

        <div className="grid" style={{ marginBottom: 16 }}>
          <div className="panel" style={{ gridColumn: "span 9" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <h3>Force‑Graph</h3>
              <div>
                <button data-testid="refresh-graph-button" className="button" onClick={() => setRefreshTick((x) => x + 1)}>Refresh</button>
              </div>
            </div>
            {error && (<div data-testid="graph-error" className="badge critical">{String(error)}</div>)}
            {loading && <div data-testid="graph-loading">Loading graph…</div>}
            {!loading && <Graph data={data} onNodeClick={(d) => console.log("node", d)} />}
          </div>
          <div className="panel" style={{ gridColumn: "span 3" }}>
            <h3>Stats</h3>
            <div className="stat" data-testid="stat-score">Score: {Math.max(0, stats.score)}</div>
            <div className="stat" data-testid="stat-projects">Projects: {stats.projects}</div>
            <div className="stat" data-testid="stat-reflections">Reflections: {stats.reflections}</div>
            <div className="stat" data-testid="stat-anomalies">Anomalies: {stats.anomalies}</div>
            <div style={{ height: 8 }} />
            <TrustMeter trust={trust} />
            <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 8 }}>
              Deception: TP {deception?.tp ?? 0} • FP {deception?.fp ?? 0} • FN {deception?.fn ?? 0} • TN {deception?.tn ?? 0}
            </div>
          </div>
        </div>

        <div className="grid" style={{ marginBottom: 16 }}>
          <div className="panel" style={{ gridColumn: "span 6" }}>
            <h3>Compute Clusters</h3>
            <label className="stat" htmlFor="clusters-k">K (2–12)</label>
            <input data-testid="clusters-k-input" id="clusters-k" className="input" type="number" min={2} max={12} step={1} value={clustersK} onChange={(e) => setClustersK(parseInt(e.target.value || "4", 10))} />
            <div style={{ height: 8 }} />
            <button data-testid="compute-clusters-button" className="button" disabled={computeBusy} onClick={handleComputeClusters}>
              {computeBusy ? "Computing…" : "Compute Clusters"}
            </button>
          </div>
          <div className="panel" style={{ gridColumn: "span 6" }}>
            <h3>Detect Anomalies</h3>
            <label className="stat" htmlFor="contamination">Contamination (0.01–0.49)</label>
            <input data-testid="contamination-input" id="contamination" className="input" type="number" min={0.01} max={0.49} step={0.01} value={contamination} onChange={(e) => setContamination(parseFloat(e.target.value || "0.05"))} />
            <div style={{ height: 8 }} />
            <button data-testid="compute-anomalies-button" className="button" disabled={computeBusy} onClick={handleComputeAnomalies}>
              {computeBusy ? "Computing…" : "Detect Anomalies"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
