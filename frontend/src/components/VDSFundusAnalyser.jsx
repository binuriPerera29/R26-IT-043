// frontend/src/components/VDSFundusAnalyser.jsx
// Cataract VDS Analyser — Light theme, production-grade UI
//cataract Function

import { useState, useRef, useEffect } from "react";
import { analyzeVDS } from "../services/api";

/* ── Grade config ─────────────────────────────────── */
const GRADE_CONFIG = {
  "Clear (No Cataract)": {
    bar: "from-emerald-400 to-teal-500",
    text: "text-emerald-600",
    pill: "bg-emerald-50 text-emerald-700 border-emerald-200",
    dot: "bg-emerald-500",
  },
  "Mild Degradation": {
    bar: "from-amber-400 to-yellow-400",
    text: "text-amber-600",
    pill: "bg-amber-50 text-amber-700 border-amber-200",
    dot: "bg-amber-400",
  },
  "Moderate Degradation": {
    bar: "from-orange-400 to-orange-500",
    text: "text-orange-600",
    pill: "bg-orange-50 text-orange-700 border-orange-200",
    dot: "bg-orange-500",
  },
  "Severe Degradation": {
    bar: "from-rose-500 to-red-500",
    text: "text-rose-600",
    pill: "bg-rose-50 text-rose-700 border-rose-200",
    dot: "bg-rose-500",
  },
};

const CLASS_ORDER  = ["No", "Mild", "Moderate", "Severe"];
const CLASS_COLORS = ["#10b981", "#f59e0b", "#f97316", "#f43f5e"];

const COMPONENT_META = [
  { key: "severity_score",  label: "Severity",      weight: "50%" },
  { key: "vessel_score",    label: "Vessel Vis.",    weight: "20%" },
  { key: "sharpness_score", label: "Sharpness",      weight: "12%" },
  { key: "contrast_score",  label: "Contrast",       weight: "8%"  },
  { key: "entropy_score",   label: "Detail Entropy", weight: "10%" },
];

/* ── Animated counter ─────────────────────────────── */
function AnimatedNumber({ value, decimals = 1 }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const duration = 900;
    const start = performance.now();
    const raf = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - t, 3);
      setDisplay(value * ease);
      if (t < 1) requestAnimationFrame(raf);
    };
    requestAnimationFrame(raf);
  }, [value]);
  return <>{display.toFixed(decimals)}</>;
}

/* ── Probability row ──────────────────────────────── */
function ProbRow({ label, value, color, index }) {
  return (
    <div className="flex items-center gap-3"
      style={{ animation: `fadeUp 0.4s ease both`, animationDelay: `${index * 70}ms` }}>
      <span className="w-20 text-xs font-medium text-slate-500 shrink-0">{label}</span>
      <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
        <div
          className="h-full transition-all duration-1000 rounded-full"
          style={{ width: `${value * 100}%`, background: color, transitionDelay: `${index * 80 + 200}ms` }}
        />
      </div>
      <span className="w-12 font-mono text-xs font-semibold text-right text-slate-600">
        <AnimatedNumber value={value * 100} decimals={1} />%
      </span>
    </div>
  );
}

/* ── Component score tile ─────────────────────────── */
function ScoreTile({ meta, value, index }) {
  return (
    <div
      className="p-4 transition-all duration-200 bg-white border rounded-xl border-slate-100 hover:border-slate-200 hover:shadow-sm"
      style={{ animation: `fadeUp 0.4s ease both`, animationDelay: `${index * 55}ms` }}
    >
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs font-medium leading-tight text-slate-400">{meta.label}</p>
        <span className="text-[9px] font-mono text-slate-300 bg-slate-50 border border-slate-100 px-1.5 py-0.5 rounded">
          {meta.weight}
        </span>
      </div>
      <p className="font-mono text-2xl font-bold tracking-tight text-slate-800">
        <AnimatedNumber value={value * 100} decimals={1} />
        <span className="text-base font-medium text-slate-400">%</span>
      </p>
      <div className="mt-2.5 h-1 rounded-full bg-slate-100 overflow-hidden">
        <div
          className="h-full transition-all duration-1000 rounded-full bg-gradient-to-r from-blue-400 to-indigo-500"
          style={{ width: `${value * 100}%`, transitionDelay: `${index * 70 + 300}ms` }}
        />
      </div>
    </div>
  );
}

/* ── Main ─────────────────────────────────────────── */
export default function VDSFundusAnalyser() {
  const [preview, setPreview]   = useState(null);
  const [file, setFile]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [result, setResult]     = useState(null);
  const [error, setError]       = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef                = useRef(null);

  function pick(f) {
    if (!f) return;
    setFile(f); setResult(null); setError(null);
    setPreview(URL.createObjectURL(f));
  }

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await analyzeVDS(file);
      setResult(data.result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setFile(null); setPreview(null); setResult(null); setError(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const cfg = result ? (GRADE_CONFIG[result.grade] ?? GRADE_CONFIG["Severe Degradation"]) : null;
  
  // Extract max probability for symmetrical display
  const maxClassProb = result?.model_probs 
    ? (Array.isArray(result.model_probs) ? Math.max(...result.model_probs) : Math.max(...Object.values(result.model_probs))) 
    : 0;

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        *, *::before, *::after { box-sizing: border-box; }

        body {
          font-family: 'Sora', sans-serif;
          background: #f6f7f9;
          color: #1e293b;
        }

        .mono { font-family: 'JetBrains Mono', monospace; }

        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }

        @keyframes blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }

        .accent-label {
          font-family: 'JetBrains Mono', monospace;
          font-size: 10px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #94a3b8;
          display: block;
        }

        .card {
          background: #ffffff;
          border: 1px solid #e8edf3;
          border-radius: 16px;
        }

        .dot-bg {
          background-color: #f6f7f9;
          background-image: radial-gradient(#d1d5db 1px, transparent 1px);
          background-size: 22px 22px;
        }

        .upload-zone {
          border: 1.5px dashed #cbd5e1;
          background: #ffffff;
          border-radius: 12px;
          cursor: pointer;
          transition: border-color 0.2s, background 0.2s;
        }
        .upload-zone:hover,
        .upload-zone.drag-over {
          border-color: #6366f1;
          background: #fafafe;
        }

        .gauge-track {
          height: 10px;
          border-radius: 99px;
          background: linear-gradient(90deg,
            rgba(16,185,129,0.15) 0%,
            rgba(245,158,11,0.15) 40%,
            rgba(249,115,22,0.15) 70%,
            rgba(244,63,94,0.15) 100%
          );
          border: 1px solid #e2e8f0;
          overflow: hidden;
        }

        .gauge-fill {
          height: 100%;
          border-radius: 99px;
          transition: width 1.1s cubic-bezier(0.22, 1, 0.36, 1);
        }

        .status-dot {
          width: 7px; height: 7px;
          border-radius: 50%;
          animation: blink 2s ease-in-out infinite;
        }

        .analyse-btn {
          flex: 1;
          height: 42px;
          border-radius: 10px;
          border: none;
          font-family: 'Sora', sans-serif;
          font-size: 13px;
          font-weight: 600;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: all 0.2s;
          cursor: pointer;
        }

        .analyse-btn:not(:disabled) {
          background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
          color: #fff;
          box-shadow: 0 4px 14px rgba(99,102,241,0.3);
        }
        .analyse-btn:not(:disabled):hover {
          box-shadow: 0 6px 20px rgba(99,102,241,0.4);
          transform: translateY(-1px);
        }
        .analyse-btn:disabled {
          background: #f1f5f9;
          color: #cbd5e1;
          cursor: not-allowed;
        }

        .reset-btn {
          height: 42px;
          width: 42px;
          border-radius: 10px;
          border: 1px solid #e2e8f0;
          background: #fff;
          color: #94a3b8;
          font-size: 16px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s;
        }
        .reset-btn:hover {
          border-color: #cbd5e1;
          color: #64748b;
        }

        .spinner {
          width: 18px; height: 18px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: #fff;
          border-radius: 50%;
          animation: spin 0.7s linear infinite;
        }

        @media (max-width: 768px) {
          .main-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>

      <div className="min-h-screen dot-bg">

        {/* ── Nav bar ── */}
        <nav style={{
          background: "rgba(255,255,255,0.9)",
          backdropFilter: "blur(12px)",
          borderBottom: "1px solid #e8edf3",
          position: "sticky", top: 0, zIndex: 10
        }}>
          <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 24px",
            display: "flex", alignItems: "center", justifyContent: "space-between", height: 56 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{
                width: 30, height: 30, borderRadius: 8,
                background: "linear-gradient(135deg, #eef2ff, #e0e7ff)",
                border: "1px solid #c7d2fe",
                display: "flex", alignItems: "center", justifyContent: "center"
              }}>
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                  stroke="#6366f1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                  <circle cx="12" cy="12" r="3"/>
                </svg>
              </div>
              <span style={{ fontWeight: 600, fontSize: 14, color: "#0f172a", letterSpacing: "-0.02em" }}>
                Fundus VDS
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", gap: 7 }}>
              <div className="status-dot" style={{ background: "#10b981" }} />
              <span className="mono" style={{ fontSize: 10, color: "#10b981", letterSpacing: "0.1em" }}>
                SYSTEM READY
              </span>
            </div>
          </div>
        </nav>

        {/* ── Body ── */}
        <div style={{ maxWidth: 1100, margin: "0 auto", padding: "36px 24px 64px" }}>

          {/* Page heading */}
          <div style={{ marginBottom: 32, animation: "fadeUp 0.5s ease both" }}>
            <span className="accent-label" style={{ marginBottom: 8 }}>Retinal Analysis Platform</span>
            <h1 style={{
              fontSize: "clamp(26px, 4vw, 34px)", fontWeight: 600,
              color: "#0f172a", letterSpacing: "-0.03em", lineHeight: 1.2
            }}>
              Visibility Degradation{" "}
              <span style={{
                background: "linear-gradient(90deg, #6366f1, #8b5cf6)",
                WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent"
              }}>Score Analyser</span>
            </h1>
            <p style={{ fontSize: 13, color: "#94a3b8", marginTop: 10, maxWidth: 460, lineHeight: 1.7 }}>
              Upload a retinal fundus photograph. Our model computes a composite VDS across 5 clinical signals to classify cataract severity.
            </p>
          </div>

          {/* Two-col grid */}
          <div className="main-grid" style={{
            display: "grid",
            gridTemplateColumns: "minmax(0,2fr) minmax(0,3fr)",
            gap: 20, alignItems: "start"
          }}>

            {/* ── LEFT ── */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

              {/* Upload card */}
              <div className="card" style={{ overflow: "hidden", animation: "fadeUp 0.5s ease both" }}>
                <div style={{
                  padding: "14px 20px", borderBottom: "1px solid #f1f5f9",
                  display: "flex", alignItems: "center", justifyContent: "space-between"
                }}>
                  <span className="accent-label" style={{ display: "inline" }}>Input Image</span>
                  {file && (
                    <span style={{
                      fontSize: 11, color: "#94a3b8",
                      overflow: "hidden", textOverflow: "ellipsis",
                      whiteSpace: "nowrap", maxWidth: 140
                    }}>{file.name}</span>
                  )}
                </div>

                {/* Drop zone */}
                <div style={{ padding: 16 }}>
                  <div
                    className={`upload-zone ${dragging ? "drag-over" : ""}`}
                    style={{ minHeight: 260 }}
                    onClick={() => inputRef.current?.click()}
                    onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files?.[0]); }}
                  >
                    <input ref={inputRef} type="file" accept="image/*"
                      style={{ display: "none" }}
                      onChange={(e) => pick(e.target.files?.[0])} />

                    {preview ? (
                      <div style={{ padding: 12 }}>
                        <img src={preview} alt="Fundus preview"
                          style={{ width: "100%", objectFit: "contain", borderRadius: 8, maxHeight: 240, display: "block" }} />
                        <div style={{
                          marginTop: 10, display: "flex", alignItems: "center",
                          gap: 6, color: "#6366f1", fontSize: 12
                        }}>
                          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
                            stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="20 6 9 17 4 12"/>
                          </svg>
                          Image ready — click Analyse to proceed
                        </div>
                      </div>
                    ) : (
                      <div style={{
                        minHeight: 260, display: "flex", flexDirection: "column",
                        alignItems: "center", justifyContent: "center",
                        padding: 32, textAlign: "center"
                      }}>
                        <div style={{
                          width: 52, height: 52, borderRadius: 14,
                          background: dragging ? "#eef2ff" : "#f8fafc",
                          border: `1.5px solid ${dragging ? "#c7d2fe" : "#e2e8f0"}`,
                          display: "flex", alignItems: "center", justifyContent: "center",
                          marginBottom: 16, transition: "all 0.2s"
                        }}>
                          <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                            stroke={dragging ? "#6366f1" : "#94a3b8"}
                            strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                          </svg>
                        </div>
                        <p style={{ fontSize: 13, fontWeight: 500, color: dragging ? "#6366f1" : "#475569", marginBottom: 4 }}>
                          {dragging ? "Release to upload" : "Drop your fundus image here"}
                        </p>
                        <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 16 }}>or click to browse files</p>
                        <div style={{ display: "flex", gap: 6 }}>
                          {["JPG", "PNG", "BMP", "TIFF"].map(f => (
                            <span key={f} className="mono" style={{
                              fontSize: 10, padding: "3px 8px", borderRadius: 5,
                              background: "#f1f5f9", border: "1px solid #e2e8f0", color: "#94a3b8"
                            }}>{f}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Actions */}
                <div style={{ padding: "0 16px 16px", display: "flex", gap: 10 }}>
                  <button className="analyse-btn" onClick={handleAnalyze} disabled={!file || loading}>
                    {loading ? (
                      <><div className="spinner" /> Analysing…</>
                    ) : (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
                          stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
                        </svg>
                        Analyse Image
                      </>
                    )}
                  </button>
                  {(preview || result) && (
                    <button className="reset-btn" onClick={handleReset}>✕</button>
                  )}
                </div>
              </div>

              {/* Error */}
              {error && (
                <div style={{
                  padding: "14px 16px", borderRadius: 12,
                  background: "#fff1f2", border: "1px solid #fecdd3",
                  color: "#be123c", fontSize: 13, display: "flex", gap: 10,
                  animation: "fadeUp 0.3s ease"
                }}>
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none"
                    stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    style={{ marginTop: 1, flexShrink: 0 }}>
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  {error}
                </div>
              )}

              {/* How it works */}
              {!result && !loading && (
                <div className="card" style={{
                  padding: "18px 20px",
                  animation: "fadeUp 0.5s ease both", animationDelay: "0.1s"
                }}>
                  <span className="accent-label" style={{ marginBottom: 14, display: "block" }}>How it works</span>
                  <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                    {[
                      ["01", "Upload a fundus photograph"],
                      ["02", "AI classifies cataract severity into 4 grades"],
                      ["03", "VDS is computed from 5 weighted signals"],
                      ["04", "Full breakdown returned in seconds"],
                    ].map(([n, t]) => (
                      <div key={n} style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
                        <span className="mono" style={{
                          fontSize: 10, color: "#a5b4fc",
                          background: "#eef2ff", border: "1px solid #c7d2fe",
                          borderRadius: 5, padding: "2px 7px", marginTop: 1, whiteSpace: "nowrap"
                        }}>{n}</span>
                        <span style={{ fontSize: 12, color: "#64748b", lineHeight: 1.65 }}>{t}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* ── RIGHT ── */}
            <div>

              {/* Empty */}
              {!result && !loading && (
                <div className="card" style={{
                  minHeight: 400, display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", textAlign: "center",
                  padding: 40, animation: "fadeUp 0.5s ease both", animationDelay: "0.15s"
                }}>
                  <div style={{
                    width: 56, height: 56, borderRadius: 16,
                    background: "#f8fafc", border: "1px solid #e2e8f0",
                    display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 16
                  }}>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
                      stroke="#cbd5e1" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                      <circle cx="12" cy="12" r="3"/>
                    </svg>
                  </div>
                  <p style={{ fontSize: 14, fontWeight: 500, color: "#94a3b8", marginBottom: 4 }}>No analysis yet</p>
                  <p style={{ fontSize: 12, color: "#cbd5e1" }}>Upload an image and click Analyse</p>
                </div>
              )}

              {/* Loading */}
              {loading && (
                <div className="card" style={{
                  minHeight: 400, display: "flex", flexDirection: "column",
                  alignItems: "center", justifyContent: "center", textAlign: "center", padding: 40
                }}>
                  <div style={{
                    width: 46, height: 46,
                    border: "2px solid #e0e7ff", borderTopColor: "#6366f1",
                    borderRadius: "50%", animation: "spin 0.8s linear infinite", marginBottom: 18
                  }} />
                  <p style={{ fontSize: 14, fontWeight: 500, color: "#475569", marginBottom: 4 }}>
                    Analysing fundus image…
                  </p>
                  <p style={{ fontSize: 12, color: "#94a3b8" }}>Running model inference</p>
                </div>
              )}

              {/* Results */}
              {result && cfg && (
                <div style={{ display: "flex", flexDirection: "column", gap: 16,
                  animation: "fadeUp 0.5s cubic-bezier(0.22,1,0.36,1) both" }}>

                  {/* VDS hero */}
                  <div className="card" style={{ padding: "24px 24px 20px" }}>
                    <div style={{
                      display: "flex", alignItems: "flex-start",
                      justifyContent: "space-between", marginBottom: 20, flexWrap: "wrap", gap: 12
                    }}>
                     

                      {/* RIGHT: PRIMARY CLASSIFICATION */}
                      <div style={{ textAlign: "right" }}>
                        <span className="accent-label" style={{ marginBottom: 6, display: "block" }}>Classification</span>
                        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "flex-end", gap: 6 }}>
                          <span className={`mono ${cfg.text}`}
                            style={{ fontSize: 42, fontWeight: 700, lineHeight: 1, letterSpacing: "-1px", textTransform: "uppercase" }}>
                            {result.predicted_class}
                          </span>
                          {maxClassProb > 0 && (
                            <span className="mono" style={{ fontSize: 18, color: "#94a3b8", fontWeight: 400 }}>
                              {(maxClassProb * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                        {result.vds_source === "seeded" && (
                          <span style={{ fontSize: 10, color: "#f59e0b", display: "block", marginTop: 4 }}>⚠ range-corrected</span>
                        )}
                      </div>
                    </div>
                    <div className="mono" style={{
                      display: "flex", justifyContent: "space-between",
                      fontSize: 9, color: "#cbd5e1", marginTop: 6, letterSpacing: "0.08em"
                    }}>
                    </div>
                  </div>

                  {/* Class probabilities */}
                  <div className="card" style={{ padding: "20px 24px" }}>
                    <span className="accent-label" style={{ marginBottom: 16, display: "block" }}>
                      Class Probabilities
                    </span>
                    <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                      {CLASS_ORDER.map((cls, i) => {
                        // Intelligent fallback: checks if the backend returned an array OR an object mapping
                        const probVal = result.model_probs 
                          ? (Array.isArray(result.model_probs) ? result.model_probs[i] : result.model_probs[cls]) 
                          : 0;
                        return (
                          <ProbRow key={cls} label={cls} value={probVal ?? 0}
                            color={CLASS_COLORS[i]} index={i} />
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
}