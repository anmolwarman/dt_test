import { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from "recharts";
import { fetchPatients, fetchCurveById, fetchCurveFromPayload } from "./api";
import "./theme.css";
import "@fontsource/inter";

function PrettyChart({ curve }) {
  const data = useMemo(() => {
    if (!curve) return [];
    const m = curve.months || [];
    const s = curve.survival || [];
    const cm = curve.cum_mortality || [];
    return m.map((x, i) => ({ month: x, survival: s[i], cum_mortality: cm[i] }));
  }, [curve]);

  return (
    <div className="card chart-card">
      <div className="section-title">OS Curve</div>
      <div style={{ width: "100%", height: "64vh", minHeight: 420 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 8, right: 12, left: 4, bottom: 8 }}>
            <CartesianGrid stroke="#2a2a35" />
            <XAxis dataKey="month" tick={{ fill: "#a9a9b6" }}
              label={{ value: "Months", fill: "#a9a9b6", position: "insideBottomRight", offset: -2 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#a9a9b6" }} tickFormatter={(v)=>`${Math.round(v*100)}%`} />
            <Tooltip contentStyle={{ background:"#17171f", border:"1px solid rgba(255,255,255,.08)", borderRadius:12 }}
                     formatter={(v)=>`${(v*100).toFixed(1)}%`} />
            <Legend />
            <Line type="monotone" dataKey="cum_mortality" name="Cumulative Mortality" stroke="#6ee7f2" strokeWidth={2.6} dot={false} />
            <Line type="monotone" dataKey="survival" name="Survival" stroke="#9b87f5" strokeWidth={2.6} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ display:"grid", gap:6 }}>
      <div className="subtle" style={{ fontSize:12, letterSpacing:".3px" }}>{label}</div>
      {children}
    </div>
  );
}

export default function OsTwinApp() {
  const [tab, setTab] = useState("existing");

  // Existing patient
  const [patients, setPatients] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [existingCurve, setExistingCurve] = useState(null);
  const [loadingExisting, setLoadingExisting] = useState(false);
  const [errExisting, setErrExisting] = useState("");

  // New patient
  const [newPatient, setNewPatient] = useState({
    patient_identifier: "NEW_PATIENT",
    age: "", primary_histology: "", extracranial_disease: "",
    kps_pre_gk: "", total_fu_time: ""
  });
  const [plans, setPlans] = useState([{ margin_dose:"", beam_on_time_min:"" }]);
  const [newCurve, setNewCurve] = useState(null);
  const [loadingNew, setLoadingNew] = useState(false);
  const [errNew, setErrNew] = useState("");

  // Load IDs (valid-only; fallback to all if none)
  useEffect(() => {
    fetchPatients(1200, true)
      .then(({ ids }) => {
        setPatients(ids || []);
        if (ids && ids.length) setSelectedId(ids[0]);
      })
      .catch(console.error);
  }, []);

  const fetchExisting = async () => {
    setErrExisting(""); setExistingCurve(null);
    if (!selectedId) return;
    setLoadingExisting(true);
    try {
      const curve = await fetchCurveById(selectedId);
      setExistingCurve(curve);
    } catch (e) {
      setErrExisting("Couldn’t fetch this patient’s curve. Try another ID.");
    } finally {
      setLoadingExisting(false);
    }
  };

  const onPlanChange = (i, k, v) => setPlans(p => { const q=[...p]; q[i] = { ...q[i], [k]: v }; return q; });
  const addPlan = () => setPlans(p => [...p, { margin_dose:"", beam_on_time_min:"" }]);
  const removePlan = (i) => setPlans(p => p.filter((_,j)=>j!==i));

  const submitNew = async () => {
    setErrNew(""); setNewCurve(null); setLoadingNew(true);
    try{
      const payload = {
        ...newPatient,
        age: newPatient.age===""? null: Number(newPatient.age),
        primary_histology: newPatient.primary_histology===""? null: Number(newPatient.primary_histology),
        extracranial_disease: newPatient.extracranial_disease===""? null: Number(newPatient.extracranial_disease),
        kps_pre_gk: newPatient.kps_pre_gk===""? null: Number(newPatient.kps_pre_gk),
        total_fu_time: newPatient.total_fu_time===""? null: Number(newPatient.total_fu_time),
        plans: plans.map(p=>({
          margin_dose: p.margin_dose===""? null: Number(p.margin_dose),
          beam_on_time_min: p.beam_on_time_min===""? null: Number(p.beam_on_time_min),
        })),
      };
      const curve = await fetchCurveFromPayload(payload);
      setNewCurve(curve);
    }catch(e){
      setErrNew("Couldn’t compute curve for this payload. Check values and try again.");
    }finally{ setLoadingNew(false); }
  };

  return (
    <div className="wrapper">
      <div className="header">
        <div className="brand">
          <div className="brand-logo" />
          <h1>BrainMets Digital Twin — OS Curve</h1>
        </div>
        <div className="toolbar">
          <button className={`btn ${tab==="existing"?"btn-primary":""}`} onClick={()=>setTab("existing")} disabled={tab==="existing"}>Existing Patient</button>
          <button className={`btn ${tab==="new"?"btn-primary":""}`} onClick={()=>setTab("new")} disabled={tab==="new"}>New Patient</button>
        </div>
      </div>

      <div className="row">
        <div className="card">
          <div className="card-title">{tab==="existing" ? "Select Patient" : "New Patient"}</div>

          {tab==="existing" ? (
            <>
              <div style={{ display:"grid", gridTemplateColumns:"1fr auto", gap:12 }}>
                <Field label="Patient ID">
                  <select className="select" value={selectedId} onChange={(e)=>setSelectedId(e.target.value)}>
                    {patients.map(id => <option key={id} value={id}>{id}</option>)}
                  </select>
                </Field>
                <div style={{ alignSelf:"end" }}>
                  <button className="btn btn-primary" onClick={fetchExisting} disabled={!selectedId || loadingExisting}>
                    {loadingExisting ? "Loading…" : "Fetch Curve"}
                  </button>
                </div>
              </div>
              {errExisting && <div className="subtle" style={{ color:"var(--danger)", marginTop:8 }}>{errExisting}</div>}
            </>
          ) : (
            <>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:12 }}>
                <Field label="Identifier">
                  <input className="input" value={newPatient.patient_identifier} onChange={(e)=>setNewPatient(p=>({...p,patient_identifier:e.target.value}))} />
                </Field>
                <Field label="Age">
                  <input className="input" type="number" value={newPatient.age} onChange={(e)=>setNewPatient(p=>({...p,age:e.target.value}))}/>
                </Field>
                <Field label="Primary Histology (numeric)">
                  <input className="input" type="number" value={newPatient.primary_histology} onChange={(e)=>setNewPatient(p=>({...p,primary_histology:e.target.value}))}/>
                </Field>
                <Field label="Extracranial Disease (0/1)">
                  <input className="input" type="number" value={newPatient.extracranial_disease} onChange={(e)=>setNewPatient(p=>({...p,extracranial_disease:e.target.value}))}/>
                </Field>
                <Field label="KPS pre-GK">
                  <input className="input" type="number" value={newPatient.kps_pre_gk} onChange={(e)=>setNewPatient(p=>({...p,kps_pre_gk:e.target.value}))}/>
                </Field>
                <Field label="Total Follow-up Time (days)">
                  <input className="input" type="number" value={newPatient.total_fu_time} onChange={(e)=>setNewPatient(p=>({...p,total_fu_time:e.target.value}))}/>
                </Field>
              </div>

              <div className="spacer" />
              <div className="section-title">Plans</div>
              <div style={{ display:"grid", gap:10 }}>
                {plans.map((r,i)=>(
                  <div key={i} className="card" style={{ padding:12 }}>
                    <div className="subtle" style={{ marginBottom:8 }}>Plan #{i+1}</div>
                    <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr auto", gap:10 }}>
                      <Field label="Margin Dose">
                        <input className="input" type="number" value={r.margin_dose} onChange={(e)=>onPlanChange(i,"margin_dose", e.target.value)}/>
                      </Field>
                      <Field label="Beam-on Time (min)">
                        <input className="input" type="number" value={r.beam_on_time_min} onChange={(e)=>onPlanChange(i,"beam_on_time_min", e.target.value)}/>
                      </Field>
                      <div style={{ alignSelf:"end" }}>
                        <button className="btn" onClick={()=>removePlan(i)} disabled={plans.length===1}>Remove</button>
                      </div>
                    </div>
                  </div>
                ))}
                <button className="btn" onClick={addPlan}>+ Add Plan</button>
              </div>

              <div className="spacer" />
              <button className="btn btn-primary" onClick={submitNew} disabled={loadingNew}>
                {loadingNew ? "Computing…" : "Predict Curve"}
              </button>
              {errNew && <div className="subtle" style={{ color:"var(--danger)", marginTop:8 }}>{errNew}</div>}
            </>
          )}
        </div>

        <div>
          <div className="subtle" style={{ margin:"8px 0 10px" }}>
            {tab==="existing" ? <>Patient: <strong>{selectedId || "—"}</strong></> : <>New Patient: <strong>{newPatient.patient_identifier || "—"}</strong></>}
          </div>
          <PrettyChart curve={tab==="existing" ? existingCurve : newCurve} />
        </div>
      </div>
    </div>
  );
}
