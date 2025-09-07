import { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ResponsiveContainer } from 'recharts';

export default function OsTwinChart() {
  const [pid, setPid] = useState("10092334");
  const [api, setApi] = useState(import.meta.env.VITE_API_URL || "http://127.0.0.1:8080");
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  async function fetchCurve() {
    setBusy(true); setErr("");
    try {
      const url = `${api.replace(/\/$/,'')}/os_curve/${encodeURIComponent(pid)}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const json = await res.json();
      const rows = json.months.map((m, i) => ({
        month: m,
        cum_mortality: json.cum_mortality[i],
        survival: json.survival[i],
      }));
      setData(rows);
    } catch(e) {
      setErr(String(e));
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{maxWidth: 960, margin: "24px auto", fontFamily: "system-ui"}}>
      <h2>Digital Twin — Overall Survival</h2>
      <div style={{display:"flex", gap:8, alignItems:"center", marginBottom: 12}}>
        <label>Patient ID:</label>
        <input value={pid} onChange={e=>setPid(e.target.value)} />
        <label>API URL:</label>
        <input style={{width:320}} value={api} onChange={e=>setApi(e.target.value)} />
        <button onClick={fetchCurve} disabled={busy}>{busy ? "Loading..." : "Fetch curve"}</button>
      </div>
      {err && <div style={{color:"crimson", marginBottom:12}}>Error: {err}</div>}
      <div style={{height: 420, border:"1px solid #eee"}}>
        <ResponsiveContainer>
          <LineChart data={data || []} margin={{ top: 20, right: 30, bottom: 20, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="month" label={{ value: 'Months since GK', position: 'insideBottom', offset: -5 }}/>
            <YAxis domain={[0,1]} label={{ value: 'Probability', angle: -90, position: 'insideLeft' }}/>
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="cum_mortality" name="Cumulative mortality" dot />
            <Line type="monotone" dataKey="survival" name="Survival" dot strokeDasharray="6 4" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}