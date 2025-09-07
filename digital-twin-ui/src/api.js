const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8080";

export async function fetchPatients(limit = 800, validOnly = true) {
  let r = await fetch(`${API}/patients?limit=${limit}&valid=${validOnly ? 1 : 0}`);
  if (!r.ok) throw new Error(`patients failed: ${r.status}`);
  let data = await r.json();
  // Fallback if backend returned empty validated list
  if ((!data.ids || data.ids.length === 0) && validOnly) {
    r = await fetch(`${API}/patients?limit=${limit}&valid=0`);
    if (!r.ok) throw new Error(`patients fallback failed: ${r.status}`);
    data = await r.json();
  }
  return data;
}

export async function fetchCurveById(patientId) {
  const r = await fetch(`${API}/os_curve/${encodeURIComponent(patientId)}`);
  if (!r.ok) throw new Error(`curve failed: ${r.status}`);
  return r.json();
}

export async function fetchCurveFromPayload(payload) {
  const r = await fetch(`${API}/os_curve_from_payload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!r.ok) throw new Error(`payload failed: ${r.status}`);
  return r.json();
}
