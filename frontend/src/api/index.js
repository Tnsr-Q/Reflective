import axios from "axios";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
if (!BACKEND_URL) {
  // eslint-disable-next-line no-console
  console.warn("REACT_APP_BACKEND_URL is not set");
}
const API = `${BACKEND_URL}/api`;

export const api = axios.create({
  baseURL: API,
});

// Projects
export async function createProject(payload) {
  const { data } = await api.post(`/projects`, payload);
  return data;
}
export async function listProjects() {
  const { data } = await api.get(`/projects`);
  return data;
}

// Reflections
export async function createReflection(payload) {
  const { data } = await api.post(`/reflections`, payload);
  return data;
}
export async function listReflections(projectId) {
  const { data } = await api.get(`/reflections`, { params: { project_id: projectId } });
  return data;
}

// Anomalies
export async function createAnomaly(payload) {
  const { data } = await api.post(`/anomalies`, payload);
  return data;
}
export async function listAnomalies(projectId) {
  const { data } = await api.get(`/anomalies`, { params: { project_id: projectId } });
  return data;
}

// Graph
export async function fetchGraph() {
  const { data } = await api.get(`/graph`);
  return data;
}

// Compute
export async function computeClusters(k = 4) {
  const { data } = await api.post(`/compute/clusters`, null, { params: { k } });
  return data;
}
export async function computeAnomalies(contamination = 0.05) {
  const { data } = await api.post(`/compute/anomalies`, null, { params: { contamination } });
  return data;
}
