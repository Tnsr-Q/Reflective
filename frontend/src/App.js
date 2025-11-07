import React, { useEffect, useMemo, useState } from "react";
import "./App.css";
import Graph from "./components/Graph";
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

  // Compute control state
  const [clustersK, setClustersK] = useState(4);
  const [contamination, setContamination] = useState(0.05);
  const [computeBusy, setComputeBusy] = useState(false);

  // Load projects
  useEffect(() => {
    (async () => {
      const list = await listProjects();
      setProjects(list);
      if (!selectedProjectId && list.length) setSelectedProjectId(list[0].id);
    })();
  }, [refreshTick]);

  const stats = useMemo(() => data?.stats || { score: 0 }, [data]);

  const handleCreateProject = async () => {
    if (!form.title.trim()) return;
    setCreating(true);
    try {
      await createProject({ title: form.title, description: form.description, status: "draft" });
      setForm({ title: "", description: "" });
      setRefreshTick((x) => x + 1);
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(e?.response?.data?.detail || e.message || "Failed to create project");
    } finally {
      setCreating(false);
    }
  };

  const handleGenerateReflection = async () => {
    if (!selectedProjectId || !prompt.trim()) return;
    setGenerating(true);
    try {
      await createReflection({ project_id: selectedProjectId, prompt });
      setPrompt("");
      setRefreshTick((x) => x + 1);
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(e?.response?.data?.detail || e.message || "Failed to generate reflection");
    } finally {
      setGenerating(false);
    }
  };

  const handleCreateAnomaly = async () => {
    if (!selectedProjectId || !anomDetail.trim()) return;
    setCreatingAnom(true);
    try {
      await createAnomaly({ project_id: selectedProjectId, detail: anomDetail, severity: anomSeverity });
      setAnomDetail("");
      setAnomSeverity("low");
      setRefreshTick((x) => x + 1);
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(e?.response?.data?.detail || e.message || "Failed to create anomaly");
    } finally {
      setCreatingAnom(false);
    }
  };

  const handleComputeClusters = async () => {
    setComputeBusy(true);
    try {
      await computeClusters(clustersK);
      setRefreshTick((x) => x + 1);
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(e?.response?.data?.detail || e.message || "Cluster compute failed");
    } finally {
      setComputeBusy(false);
    }
  };

  const handleComputeAnomalies = async () => {
    setComputeBusy(true);
    try {
      await computeAnomalies(contamination);
      setRefreshTick((x) => x + 1);
    } catch (e) {
      // eslint-disable-next-line no-alert
      alert(e?.response?.data?.detail || e.message || "Anomaly detection failed");
    } finally {
      setComputeBusy(false);
    }
  };

  return (
    <div className="App">
      <header className="header">
        <h1 data-testid="app-title">AI‑Reflect Dashboard</h1>
        <p>Projects, Reflections (AI), Clusters & Anomalies mapped as a force‑graph</p>
      </header>

      <main className="container" style={{ maxWidth: 1200, margin: "0 auto", padding: 16 }}>
        <div className="grid" style={{ marginBottom: 16 }}>
          <div className="panel" style={{ gridColumn: "span 4" }}>
            <h3>Create Project</h3>
            <input
              data-testid="project-title-input"
              className="input"
              placeholder="Project title"
              value={form.title}
              onChange={(e) => setForm((f) => ({ ...f, title: e.target.value }))}
            />
            <div style={{ height: 8 }} />
            <textarea
              data-testid="project-description-textarea"
              className="textarea"
              placeholder="Description (optional)"
              rows={3}
              value={form.description}
              onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
            />
            <div style={{ height: 8 }} />
            <button
              data-testid="create-project-button"
              className="button"
              disabled={creating || !form.title.trim()}
              onClick={handleCreateProject}
            >
              {creating ? "Creating…" : "Create Project"}
            </button>
          </div>

          <div className="panel" style={{ gridColumn: "span 4" }}>
            <h3>Generate Reflection (AI)</h3>
            <select
              data-testid="project-select"
              className="select"
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
            >
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.title}</option>
              ))}
            </select>
            <div style={{ height: 8 }} />
            <textarea
              data-testid="reflection-prompt-textarea"
              className="textarea"
              placeholder="Enter context/prompt for reflection (2-4 sentences will be generated)."
              rows={3}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
            />
            <div style={{ height: 8 }} />
            <button
              data-testid="generate-reflection-button"
              className="button"
              disabled={generating || !selectedProjectId || !prompt.trim()}
              onClick={handleGenerateReflection}
            >
              {generating ? "Generating…" : "Generate Reflection"}
            </button>
          </div>

          <div className="panel" style={{ gridColumn: "span 4" }}>
            <h3>Log Anomaly</h3>
            <select
              data-testid="anomaly-project-select"
              className="select"
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
            >
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.title}</option>
              ))}
            </select>
            <div style={{ height: 8 }} />
            <textarea
              data-testid="anomaly-detail-textarea"
              className="textarea"
              placeholder="Describe the anomaly"
              rows={3}
              value={anomDetail}
              onChange={(e) => setAnomDetail(e.target.value)}
            />
            <div style={{ height: 8 }} />
            <select
              data-testid="anomaly-severity-select"
              className="select"
              value={anomSeverity}
              onChange={(e) => setAnomSeverity(e.target.value)}
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
              <option value="critical">Critical</option>
            </select>
            <div style={{ height: 8 }} />
            <button
              data-testid="create-anomaly-button"
              className="button"
              disabled={creatingAnom || !selectedProjectId || !anomDetail.trim()}
              onClick={handleCreateAnomaly}
            >
              {creatingAnom ? "Saving…" : "Create Anomaly"}
            </button>
          </div>
        </div>

        <div className="grid" style={{ marginBottom: 16 }}>
          <div className="panel" style={{ gridColumn: "span 9" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
              <h3>Force‑Graph</h3>
              <div>
                <button
                  data-testid="refresh-graph-button"
                  className="button"
                  onClick={() => setRefreshTick((x) => x + 1)}
                >
                  Refresh
                </button>
              </div>
            </div>
            {error && (
              <div data-testid="graph-error" className="badge critical">{String(error)}</div>
            )}
            {loading && <div data-testid="graph-loading">Loading graph…</div>}
            {!loading && <Graph data={data} onNodeClick={(d) => console.log("node", d)} />}
          </div>

          <div className="panel" style={{ gridColumn: "span 3" }}>
            <h3>Stats</h3>
            <div className="stat" data-testid="stat-score">Score: {stats.score}</div>
            <div className="stat" data-testid="stat-projects">Projects: {stats.projects}</div>
            <div className="stat" data-testid="stat-reflections">Reflections: {stats.reflections}</div>
            <div className="stat" data-testid="stat-anomalies">Anomalies: {stats.anomalies}</div>
            <div style={{ marginTop: 8, fontSize: 12, color: "#94a3b8" }}>
              Tip: Score is 2×reflections − anomalies (for quick visual feedback)
            </div>
          </div>
        </div>

        <div className="grid" style={{ marginBottom: 16 }}>
          <div className="panel" style={{ gridColumn: "span 6" }}>
            <h3>Compute Clusters</h3>
            <label className="stat" htmlFor="clusters-k">K (2–12)</label>
            <input
              data-testid="clusters-k-input"
              id="clusters-k"
              className="input"
              type="number"
              min={2}
              max={12}
              step={1}
              value={clustersK}
              onChange={(e) => setClustersK(parseInt(e.target.value || "4", 10))}
            />
            <div style={{ height: 8 }} />
            <button
              data-testid="compute-clusters-button"
              className="button"
              disabled={computeBusy}
              onClick={handleComputeClusters}
            >
              {computeBusy ? "Computing…" : "Compute Clusters"}
            </button>
          </div>

          <div className="panel" style={{ gridColumn: "span 6" }}>
            <h3>Detect Anomalies</h3>
            <label className="stat" htmlFor="contamination">Contamination (0.01–0.49)</label>
            <input
              data-testid="contamination-input"
              id="contamination"
              className="input"
              type="number"
              min={0.01}
              max={0.49}
              step={0.01}
              value={contamination}
              onChange={(e) => setContamination(parseFloat(e.target.value || "0.05"))}
            />
            <div style={{ height: 8 }} />
            <button
              data-testid="compute-anomalies-button"
              className="button"
              disabled={computeBusy}
              onClick={handleComputeAnomalies}
            >
              {computeBusy ? "Computing…" : "Detect Anomalies"}
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
