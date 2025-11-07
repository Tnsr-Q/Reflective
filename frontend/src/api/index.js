import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
export const api = axios.create({ baseURL: API });

// Existing endpoints
export async function createProject(payload) { const { data } = await api.post(`/projects`, payload); return data; }
export async function listProjects() { const { data } = await api.get(`/projects`); return data; }
export async function createReflection(payload) { const { data } = await api.post(`/reflections`, payload); return data; }
export async function listReflections(projectId) { const { data } = await api.get(`/reflections`, { params: { project_id: projectId } }); return data; }
export async function createAnomaly(payload) { const { data } = await api.post(`/anomalies`, payload); return data; }
export async function listAnomalies(projectId) { const { data } = await api.get(`/anomalies`, { params: { project_id: projectId } }); return data; }
export async function fetchGraph() { const { data } = await api.get(`/graph`); return data; }
export async function computeClusters(k = 4) { const { data } = await api.post(`/compute/clusters`, null, { params: { k } }); return data; }
export async function computeAnomalies(contamination = 0.05) { const { data } = await api.post(`/compute/anomalies`, null, { params: { contamination } }); return data; }
export async function getTicker() { const { data } = await api.get(`/ticker`); return data; }
export async function getSchedulerStatus() { const { data } = await api.get(`/scheduler/status`); return data; }
export async function getLatestPredictions() { const { data } = await api.get(`/predictions/latest`); return data; }
export async function getPredictionHistory(horizon, limit = 50) { const { data } = await api.get(`/predictions/history`, { params: { horizon, limit } }); return data; }

// New: misinfo & metrics
export async function injectMisinfo(headline) { const { data } = await api.post(`/misinfo/inject`, null, { params: { headline } }); return data; }
export async function fetchMisinfoLatest(limit = 5) { const { data } = await api.get(`/misinfo/latest`, { params: { limit } }); return data; }
export async function runOpponent(persona = 'honest', horizon = '1h') { const { data } = await api.post(`/opponent/predictions`, null, { params: { persona, horizon } }); return data; }
export async function fetchTrust() { const { data } = await api.get(`/metrics/trust`); return data; }
export async function fetchDeceptionMetrics() { const { data } = await api.get(`/metrics/deception`); return data; }
