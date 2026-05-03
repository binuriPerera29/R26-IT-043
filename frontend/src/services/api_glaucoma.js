// src/services/api.js
// All API calls live here. Components never call fetch() directly.

const BASE_URL = "http://localhost:5004/api";

async function handleResponse(res) {
  const data = await res.json().catch(() => ({ error: "Invalid JSON response" }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// Health
export async function checkHealth() {
  const res = await fetch(`${BASE_URL}/health`);
  return handleResponse(res);
}

// Module 1 — Glaucoma Classification
export async function predictGlaucoma(imageFile, ttaSteps = 5) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("tta_steps", ttaSteps);
  const res = await fetch(`${BASE_URL}/glaucoma/predict`, { method: "POST", body: form });
  return handleResponse(res);
}

export async function getGlaucomaClasses() {
  const res = await fetch(`${BASE_URL}/glaucoma/classes`);
  return handleResponse(res);
}

// Module 2 — CDR Analysis
export async function analyseCDR(imageFile) {
  const form = new FormData();
  form.append("image", imageFile);
  const res = await fetch(`${BASE_URL}/cdr/analyse`, { method: "POST", body: form });
  return handleResponse(res);
}

// Module 3 — placeholder
// export async function predictModule3(data) { ... }

// Module 4 — placeholder
// export async function predictModule4(data) { ... }