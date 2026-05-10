/**
 * GlaucomaFundusAnalyser.jsx
 * AI-Powered Glaucoma & CDR Structural Analysis System
 * Pure Tailwind CSS + Background Dot Design
 */

import { useState, useRef, useCallback } from "react";
import { predictGlaucoma, analyseCDR } from "../services/api_glaucoma";

// ── Metadata ──────────────────────────────────────────────────────────────
const RISK_META = {
  normal: {
    color: "text-emerald-500",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    dot: "bg-emerald-500",
    bar: "bg-emerald-500",
    accent: "border-l-emerald-500",
    badge: "bg-emerald-50 text-emerald-500 border-emerald-200",
    label: "Normal / Low Risk",
  },
  early: {
    color: "text-orange-500",
    bg: "bg-orange-50",
    border: "border-orange-200",
    dot: "bg-orange-500",
    bar: "bg-orange-500",
    accent: "border-l-orange-500",
    badge: "bg-orange-50 text-orange-500 border-orange-200",
    label: "Early Glaucoma",
  },
  advanced: {
    color: "text-red-500",
    bg: "bg-red-50",
    border: "border-red-200",
    dot: "bg-red-500",
    bar: "bg-red-500",
    accent: "border-l-red-500",
    badge: "bg-red-50 text-red-500 border-red-200",
    label: "Advanced Glaucoma",
  },
};

// ── Components ─────────────────────────────────────────────────────────────

function EyeIcon({ size = 18, className = "" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function CDRGauge({ cdr }) {
  const val = cdr ?? 0;
  const angle = Math.PI - val * Math.PI;
  const nx = Math.cos(angle);
  const ny = Math.sin(angle);

  const arcPath = (from, to, rInner, rOuter) => {
    const steps = 40;
    const pts = [];
    for (let i = 0; i <= steps; i++) {
      const t = Math.PI - (from + (to - from) * (i / steps)) * Math.PI;
      pts.push(`${Math.cos(t) * rOuter},${-Math.sin(t) * rOuter}`);
    }
    for (let i = steps; i >= 0; i--) {
      const t = Math.PI - (from + (to - from) * (i / steps)) * Math.PI;
      pts.push(`${Math.cos(t) * rInner},${-Math.sin(t) * rInner}`);
    }
    return `M ${pts[0]} L ${pts.slice(1).join(" L ")} Z`;
  };

  return (
    <div className="flex flex-col items-center justify-center p-4">
      <svg viewBox="-1.3 -1.15 2.6 1.55" className="w-full max-w-[200px]">
        <path d={arcPath(0, 0.4, 0.6, 0.9)} fill="#10b981" opacity="0.8" />
        <path d={arcPath(0.4, 0.7, 0.6, 0.9)} fill="#f59e0b" opacity="0.8" />
        <path d={arcPath(0.7, 1.0, 0.6, 0.9)} fill="#ef4444" opacity="0.8" />
        <text x="0" y="0.4" textAnchor="middle" fontSize="0.32" fontWeight="800" className="font-mono fill-slate-900">
          {cdr !== null && cdr !== undefined ? cdr.toFixed(2) : "—"}
        </text>
        <text x="0" y="-0.4" textAnchor="middle" fontSize="0.1" className="font-mono tracking-widest uppercase fill-slate-400">CDR RATIO</text>
        {(cdr !== null && cdr !== undefined) && (
          <>
            <line x1="0" y1="0" x2={nx * 0.75} y2={-ny * 0.75} stroke="#1e293b" strokeWidth="0.05" strokeLinecap="round" />
            <circle cx="0" cy="0" r="0.08" fill="#fff" stroke="#cbd5e1" strokeWidth="0.02" />
          </>
        )}
      </svg>
    </div>
  );
}

// ── Main Analyser ──────────────────────────────────────────────────────────

export default function GlaucomaFundusAnalyser() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef(null);

  // Define the dot pattern style object
  const dotBackgroundStyle = {
    backgroundColor: "#f6f7f9",
    backgroundImage: "radial-gradient(#d1d5db 1px, transparent 1px)",
    backgroundSize: "22px 22px",
  };

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleAnalyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const [glaucomaData, cdrData] = await Promise.all([
        predictGlaucoma(file, 5),
        analyseCDR(file)
      ]);
      setResult({ glaucoma: glaucomaData, cdr: cdrData });
    } catch (err) {
      setError("Analysis failed. Please ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const activeMeta = result ? RISK_META[result.glaucoma.prediction.class_name] || RISK_META.normal : null;

  return (
    <div className="w-full min-h-screen font-sans text-slate-900" style={dotBackgroundStyle}>


      <main className="max-w-[1200px] mx-auto px-6 py-10">
        {/* Header */}
        <div className="mb-10 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-slate-900">Retinal Glaucoma Screening</h1>
          <p className="text-sm text-slate-500">Automated Optic Disc segmentation and diagnostic classification</p>
        </div>

        {/* Upload Area */}
        {!result && (
          <div 
            className={`max-w-xl mx-auto border-2 border-dashed rounded-3xl p-10 transition-all bg-white shadow-sm
              ${isDragging ? "border-blue-400 bg-blue-50/50" : "border-slate-200 hover:border-blue-300"}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); }}
            onClick={() => !loading && inputRef.current?.click()}
          >
            <input ref={inputRef} type="file" className="hidden" onChange={(e) => handleFile(e.target.files[0])} />
            
            {preview ? (
              <div className="flex flex-col items-center">
                <img src={preview} alt="Preview" className="object-cover w-48 h-48 mb-4 border shadow-sm rounded-2xl border-slate-100" />
                <p className="mb-6 font-mono text-xs text-slate-400">{file.name}</p>
                <div className="flex flex-col items-center gap-3 sm:flex-row">
                  <button
                    onClick={(e) => { e.stopPropagation(); handleAnalyse(); }}
                    disabled={loading}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-2.5 rounded-xl text-sm font-bold shadow-lg shadow-blue-200 transition-all disabled:opacity-50"
                  >
                    {loading ? "Processing..." : "Start Analysis"}
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      inputRef.current?.click();
                    }}
                    disabled={loading}
                    className="px-8 py-2.5 rounded-xl text-sm font-bold text-slate-700 bg-white border border-slate-200 shadow-sm hover:bg-slate-50 transition-all disabled:opacity-50"
                  >
                    Upload Another Image
                  </button>
                </div>
              </div>
            ) : (
              <div className="text-center cursor-pointer">
                <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 border bg-slate-50 rounded-2xl border-slate-100">
                  <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" /></svg>
                </div>
                <p className="text-sm font-bold text-slate-700">Click to upload fundus image</p>
                <p className="mt-1 text-xs text-slate-400">Supports High-res JPG, PNG</p>
              </div>
            )}
          </div>
        )}

        {/* Results View */}
        {result && (
          <div className="grid grid-cols-1 gap-6 duration-700 lg:grid-cols-12 animate-in fade-in slide-in-from-bottom-4">
            
            {/* Left: Visual Analysis */}
            <div className="flex flex-col gap-6 lg:col-span-4">
              <div className="p-4 bg-white border shadow-sm border-slate-200 rounded-3xl">
                <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-3">Segmented Fundus</span>
                <div className="relative overflow-hidden aspect-square rounded-2xl bg-slate-100 group">
                  <img src={result.cdr.overlay_image || preview} alt="Analysis" className="object-cover w-full h-full" />
                  <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md text-white text-[9px] px-2 py-1 rounded-full font-bold">SEGMENTED</div>
                </div>
              </div>

              
            </div>

            {/* Right: Clinical Metrics */}
            <div className="flex flex-col gap-6 lg:col-span-8">
              
              {/* Prediction Banner */}
              <div className={`p-6 rounded-3xl border shadow-sm ${activeMeta.bg} ${activeMeta.border} flex items-center justify-between`}>
                <div>
                  <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-1">AI Classification</span>
                  <h2 className={`text-3xl font-black ${activeMeta.color}`}>{activeMeta.label}</h2>
                  <p className="mt-1 text-xs font-medium text-slate-500">Based on global features and structural metrics</p>
                </div>
                <div className="text-right">
                   <div className={`text-4xl font-black font-mono ${activeMeta.color}`}>{result.glaucoma.prediction.confidence}%</div>
                   <div className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">Confidence</div>
                </div>
              </div>

              <div className="flex justify-end">
                <button
                  onClick={handleReset}
                  className="px-5 py-3 text-sm font-semibold text-slate-700 bg-white border border-slate-200 rounded-full shadow-sm hover:bg-slate-50 transition"
                >
                  Upload Another Image
                </button>
              </div>

              <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
                {/* Probabilities */}
                {/* <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
                  <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-6">Class Probabilities</span>
                  <div className="space-y-5">
                    {Object.entries(result.glaucoma.prediction.probabilities).map(([key, val]) => {
                      const meta = RISK_META[key] || RISK_META.normal;
                      return (
                        <div key={key}>
                          <div className="flex justify-between items-center mb-1.5">
                            <span className="text-xs font-bold text-slate-700">{key}</span>
                            <span className="font-mono text-xs font-bold text-slate-400">{val}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-slate-50 rounded-full overflow-hidden">
                            <div 
                              className={`h-full ${meta.bar} transition-all duration-1000`} 
                              style={{ width: `${val}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div> */}
              </div>             

            </div>
          </div>
        )}
      </main>
      
      {error && (
        <div className="fixed px-4 py-2 text-xs font-bold text-red-600 -translate-x-1/2 border border-red-200 rounded-lg shadow-xl bottom-6 left-1/2 bg-red-50">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}