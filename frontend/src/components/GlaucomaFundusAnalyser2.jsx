/**
 * GlaucomaFundusAnalyser.jsx
 * AI-Powered Glaucoma & CDR Structural Analysis System
 * Pure Tailwind CSS + Background Dot Design
 * — with Grad-CAM heatmap panel (notebook: R26-IT-043 | Chavindee M.A.P)
 * LAYOUT FIX: balanced 5/7 col split, Grad-CAM summary moved to right column,
 *             mid-row 2-col grid, navbar z-index corrected.
 * OOD UPDATE: when the backend flags an image as out-of-distribution
 *             (prediction.class_name === "ood" / ood.is_ood === true),
 *             the classification label, probabilities, Grad-CAM panels,
 *             Segmented Fundus, CDR Gauge, and structural-metrics-from-
 *             classification are hidden. The original uploaded image and
 *             a dedicated OOD details card are always shown instead.
 */

import { useState, useRef, useCallback } from "react";
import { predictGlaucoma, analyseCDR } from "../services/api_glaucoma";

// ── Metadata ──────────────────────────────────────────────────────────────
const RISK_META = {
  Normal: {
    color: "text-emerald-500",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    dot: "bg-emerald-500",
    bar: "bg-emerald-500",
    badge: "bg-emerald-50 text-emerald-500 border-emerald-200",
    label: "Normal / Low Risk",
  },
  Suspect: {
    color: "text-orange-500",
    bg: "bg-orange-50",
    border: "border-orange-200",
    dot: "bg-orange-500",
    bar: "bg-orange-500",
    badge: "bg-orange-50 text-orange-500 border-orange-200",
    label: "Glaucoma Suspect",
  },
  Glaucoma: {
    color: "text-red-500",
    bg: "bg-red-50",
    border: "border-red-200",
    dot: "bg-red-500",
    bar: "bg-red-500",
    badge: "bg-red-50 text-red-500 border-red-200",
    label: "Glaucoma Detected",
  },
  // NEW — used whenever the image is flagged out-of-distribution
  // (Mahalanobis check) or the classifier's confidence is below 70%.
  ood: {
    color: "text-slate-500",
    bg: "bg-slate-50",
    border: "border-slate-200",
    dot: "bg-slate-400",
    bar: "bg-slate-400",
    badge: "bg-slate-100 text-slate-500 border-slate-200",
    label: "Unable to Classify Reliably",
  },
};

const GRADCAM_TABS = [
  { key: "overlay_box", label: "Overlay + Box" },
  { key: "overlay", label: "Overlay" },
  { key: "heatmap", label: "Heatmap" },
];

const OOD_REASON_TEXT = {
  mahalanobis_distance: "Image features are unfamiliar to the model (out-of-distribution).",
  low_confidence: "The model's prediction confidence was too low to trust.",
  mahalanobis_and_low_confidence:
    "Image features are unfamiliar to the model AND prediction confidence was too low.",
};

// ── Components ─────────────────────────────────────────────────────────────

function EyeIcon({ size = 18, className = "" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function AlertTriangleIcon({ size = 18, className = "" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
      <line x1="12" y1="9" x2="12" y2="13" />
      <line x1="12" y1="17" x2="12.01" y2="17" />
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
    <div className="flex flex-col items-center justify-center pt-2 pb-1">
      <svg viewBox="-1.3 -1.15 2.6 1.55" className="w-full max-w-[190px]">
        <path d={arcPath(0, 0.4, 0.6, 0.9)} fill="#10b981" opacity="0.85" />
        <path d={arcPath(0.4, 0.6, 0.6, 0.9)} fill="#f59e0b" opacity="0.85" />
        <path d={arcPath(0.6, 1.0, 0.6, 0.9)} fill="#ef4444" opacity="0.85" />
        <text
          x="0"
          y="0.38"
          textAnchor="middle"
          fontSize="0.3"
          fontWeight="800"
          className="font-mono fill-slate-900"
        >
          {cdr !== null && cdr !== undefined ? cdr.toFixed(2) : "—"}
        </text>
        <text
          x="0"
          y="-0.42"
          textAnchor="middle"
          fontSize="0.09"
          className="font-mono tracking-widest uppercase fill-slate-400"
        >
          CDR RATIO
        </text>
        {cdr !== null && cdr !== undefined && (
          <>
            <line
              x1="0"
              y1="0"
              x2={nx * 0.75}
              y2={-ny * 0.75}
              stroke="#1e293b"
              strokeWidth="0.05"
              strokeLinecap="round"
            />
            <circle
              cx="0"
              cy="0"
              r="0.08"
              fill="#fff"
              stroke="#cbd5e1"
              strokeWidth="0.02"
            />
          </>
        )}
      </svg>
      {/* Range legend */}
      <div className="grid grid-cols-3 gap-1.5 w-full mt-3">
        {[
          { label: "Normal", range: "0–0.4", bg: "bg-emerald-50", text: "text-emerald-600", sub: "text-emerald-700" },
          { label: "Suspect", range: "0.4–0.6", bg: "bg-orange-50", text: "text-orange-500", sub: "text-orange-700" },
          { label: "High Risk", range: "0.6–1.0", bg: "bg-red-50", text: "text-red-500", sub: "text-red-700" },
        ].map((r) => (
          <div key={r.label} className={`${r.bg} rounded-xl p-2 text-center`}>
            <div className={`font-bold text-[9px] uppercase tracking-wide ${r.text}`}>{r.label}</div>
            <div className={`font-mono text-[11px] font-bold ${r.sub}`}>{r.range}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * OriginalImagePanel — always shown, regardless of OOD status.
 */
function OriginalImagePanel({ preview }) {
  return (
    <div className="p-4 bg-white border shadow-sm border-slate-200 rounded-3xl">
      <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-3">
        Original Uploaded Image
      </span>
      <div className="relative overflow-hidden aspect-square rounded-2xl bg-slate-100">
        <img src={preview} alt="Original upload" className="object-cover w-full h-full" />
        <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md text-white text-[9px] px-2 py-1 rounded-full font-bold">
          ORIGINAL
        </div>
      </div>
    </div>
  );
}

/**
 * OODDetailsCard — right column card explaining why the image was flagged.
 * Always shown when the image is out-of-distribution / low-confidence.
 */
function OODDetailsCard({ ood }) {
  if (!ood) return null;

  const reasonText =
    OOD_REASON_TEXT[ood.reason] ||
    "The classifier could not produce a reliable diagnosis for this image.";

  return (
    <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
      <div className="flex items-center gap-2 mb-4">
        <AlertTriangleIcon size={16} className="text-slate-400" />
        <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest">
          Out-of-Distribution Detection
        </span>
      </div>

      <p className="mb-4 text-sm font-medium leading-relaxed text-slate-600">{reasonText}</p>

      <div className="grid grid-cols-2 gap-3">
        <div className="p-3 text-center bg-slate-50 rounded-2xl">
          <div className="font-mono text-base font-black text-slate-700">
            {ood.mahalanobis_distance != null ? ood.mahalanobis_distance : "—"}
          </div>
          <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mt-0.5">
            Distance Score
          </div>
        </div>
        <div className="p-3 text-center bg-slate-50 rounded-2xl">
          <div className="font-mono text-base font-black text-slate-700">
            {ood.mahalanobis_threshold != null ? ood.mahalanobis_threshold : "—"}
          </div>
          <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mt-0.5">
            Max. Allowed
          </div>
        </div>
      </div>

      {ood.mahalanobis_check_enabled === false && (
        <p className="mt-4 text-[10px] font-medium text-slate-400">
          Note: out-of-distribution feature check is disabled on the backend — this flag is
          based on prediction confidence only.
        </p>
      )}
    </div>
  );
}

/**
 * GradCAMPanel — left column image viewer with tabs
 */
function GradCAMPanel({ gradcam, className_pred }) {
  const [activeTab, setActiveTab] = useState("overlay_box");

  if (!gradcam) {
    return (
      <div className="p-5 bg-white border shadow-sm border-slate-200 rounded-3xl">
        <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-2">
          Grad-CAM Heatmap
        </span>
        <div className="flex items-center justify-center h-32 text-xs text-slate-400">
          Grad-CAM not available
        </div>
      </div>
    );
  }

  const activeSrc = gradcam.images[activeTab];

  return (
    <div className="p-5 bg-white border shadow-sm border-slate-200 rounded-3xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest">
          Grad-CAM · backbone.conv_head
        </span>
        <span className="font-mono text-[9px] text-slate-300 uppercase">
          Target: {className_pred}
        </span>
      </div>

      {/* Tab switcher */}
      <div className="flex gap-1 p-1 mb-3 bg-slate-50 rounded-xl">
        {GRADCAM_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex-1 text-[10px] font-bold py-1 rounded-lg transition-all ${
              activeTab === tab.key
                ? "bg-white text-slate-800 shadow-sm"
                : "text-slate-400 hover:text-slate-600"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Image */}
      <div className="relative overflow-hidden rounded-2xl bg-slate-100 aspect-square">
        <img src={activeSrc} alt={activeTab} className="object-cover w-full h-full" />
        <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-md text-white text-[9px] px-2 py-1 rounded-full font-bold uppercase">
          {activeTab.replace("_", " ")}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-2 gap-2 mt-3">
        {[
          { label: "Peak Activation", val: `${gradcam.activation_peak}%` },
          { label: "Mean Activation", val: `${gradcam.activation_mean}%` },
          { label: "Target Layer", val: "conv_head" },
          { label: "High-Act Box", val: gradcam.high_activation_box ? "Detected" : "None" },
        ].map((s, i) => (
          <div key={i} className="bg-slate-50 rounded-xl p-2.5">
            <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mb-0.5">
              {s.label}
            </div>
            <div className="font-mono text-xs font-bold text-slate-800">{s.val}</div>
          </div>
        ))}
      </div>

      {/* Colour scale */}
      <div className="mt-3">
        <div className="flex justify-between mb-1">
          <span className="font-mono text-[9px] text-slate-400">Low</span>
          <span className="font-mono text-[9px] text-slate-400">Activation Intensity</span>
          <span className="font-mono text-[9px] text-slate-400">High</span>
        </div>
        <div
          className="h-2 rounded-full"
          style={{
            background:
              "linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)",
          }}
        />
      </div>
    </div>
  );
}

/**
 * GradCAMSummaryCard — right column explainability card
 */
function GradCAMSummaryCard({ gradcam }) {
  if (!gradcam) return null;
  return (
    <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
      <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-4">
        Grad-CAM Explainability
      </span>
      <div className="grid grid-cols-3 gap-3 mb-3">
        {[
          { label: "Peak Activation", val: `${gradcam.activation_peak}%`, color: "text-orange-500" },
          { label: "Mean Activation", val: `${gradcam.activation_mean}%`, color: "text-blue-500" },
          { label: "Target Layer", val: "conv_head", color: "text-purple-500" },
        ].map((s, i) => (
          <div key={i} className="p-3 text-center bg-slate-50 rounded-2xl">
            <div className={`font-mono text-base font-black ${s.color}`}>{s.val}</div>
            <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mt-0.5">
              {s.label}
            </div>
          </div>
        ))}
      </div>
      {gradcam.high_activation_box && (
        <p className="text-xs leading-relaxed text-slate-500">
          High-activation region detected at coordinates ({gradcam.high_activation_box.x},{" "}
          {gradcam.high_activation_box.y}) — size {gradcam.high_activation_box.width} ×{" "}
          {gradcam.high_activation_box.height} px. The model focused on this optic disc region
          to make its prediction.
        </p>
      )}
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
        analyseCDR(file),
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

  // NEW — true when the backend flagged this image as out-of-distribution
  // OR the classifier's confidence was below the 70% threshold.
  const isOOD = result
    ? result.glaucoma?.prediction?.class_name === "ood" || result.glaucoma?.ood?.is_ood === true
    : false;

  const activeMeta = result
    ? isOOD
      ? RISK_META.ood
      : RISK_META[result.glaucoma.prediction.class_name] || RISK_META.Normal
    : null;

  return (
    <div className="w-full min-h-screen font-sans text-slate-900" style={dotBackgroundStyle}>

      {/* ── Main ── */}
      <main className="max-w-[1400px] mx-auto px-6 py-10">

        {/* Page Header */}
        <div className="mb-10 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-slate-900">
            AI-Assisted Glaucoma Screening
          </h1>
          <p className="text-sm text-slate-500">
            Automated Optic Disc segmentation, diagnostic classification and Grad-CAM
            explainability
          </p>
        </div>

        {/* ── Upload Area ── */}
        {!result && (
          <div
            className={`max-w-xl mx-auto border-2 border-dashed rounded-3xl p-10 transition-all bg-white shadow-sm
              ${isDragging ? "border-blue-400 bg-blue-50/50" : "border-slate-200 hover:border-blue-300"}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              handleFile(e.dataTransfer.files[0]);
            }}
            onClick={() => !loading && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />
            {preview ? (
              <div className="flex flex-col items-center">
                <img
                  src={preview}
                  alt="Preview"
                  className="object-cover w-48 h-48 mb-4 border shadow-sm rounded-2xl border-slate-100"
                />
                <p className="mb-6 font-mono text-xs text-slate-400">{file.name}</p>
                <button
                  onClick={(e) => { e.stopPropagation(); handleAnalyse(); }}
                  disabled={loading}
                  className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-2.5 rounded-xl text-sm font-bold shadow-lg shadow-blue-200 transition-all disabled:opacity-50"
                >
                  {loading ? "Processing..." : "Start Analysis"}
                </button>
              </div>
            ) : (
              <div className="text-center cursor-pointer">
                <div className="flex items-center justify-center w-16 h-16 mx-auto mb-4 border bg-slate-50 rounded-2xl border-slate-100">
                  <svg className="w-6 h-6 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                  </svg>
                </div>
                <p className="text-sm font-bold text-slate-700">Click to upload fundus image</p>
                <p className="mt-1 text-xs text-slate-400">Supports retinal fundus images in JPG and PNG formats</p>
              </div>
            )}
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-[5fr_7fr] gap-6 animate-in fade-in slide-in-from-bottom-4 duration-700">

            {/* ── LEFT COLUMN: image + CDR + Grad-CAM viewer ── */}
            <div className="flex flex-col gap-5">

              {/* Original uploaded image — ALWAYS shown, OOD or not */}
              <OriginalImagePanel preview={preview} />

              {/* Segmented Fundus + CDR Gauge — hidden when OOD, since both
                  come from the CDR pipeline and shouldn't be shown alongside
                  an unreliable prediction */}
              {!isOOD && (
                <>
                  {/* Segmented Fundus (independent CDR pipeline) */}
                  <div className="p-4 bg-white border shadow-sm border-slate-200 rounded-3xl">
                    <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-3">
                      Segmented Fundus
                    </span>
                    <div className="relative overflow-hidden aspect-square rounded-2xl bg-slate-100">
                      <img
                        src={result.cdr.overlay_image || preview}
                        alt="Segmented"
                        className="object-cover w-full h-full"
                      />
                      <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-md text-white text-[9px] px-2 py-1 rounded-full font-bold">
                        SEGMENTED
                      </div>
                    </div>
                  </div>

                  {/* CDR Gauge */}
                  <div className="p-5 bg-white border shadow-sm border-slate-200 rounded-3xl">
                    <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-2">
                      Cup-to-Disc Ratio
                    </span>
                    <CDRGauge cdr={result.cdr.cdr.value} />
                  </div>
                </>
              )}

              {/* Grad-CAM image panel — hidden when OOD (tied to an unreliable prediction) */}
              {!isOOD && (
                <GradCAMPanel
                  gradcam={result.glaucoma.gradcam}
                  className_pred={result.glaucoma.prediction.class_name}
                />
              )}
            </div>

            {/* ── RIGHT COLUMN: classification + metrics ── */}
            <div className="flex flex-col gap-5">

              {isOOD ? (
                <>
                  {/* OOD Banner — no class label / confidence-as-diagnosis shown */}
                  <div
                    className={`p-6 rounded-3xl border shadow-sm ${activeMeta.bg} ${activeMeta.border} flex items-center gap-4`}
                  >
                    <div className="flex items-center justify-center w-12 h-12 bg-white border rounded-2xl border-slate-200 shrink-0">
                      <AlertTriangleIcon size={22} className="text-slate-400" />
                    </div>
                    <div>
                      <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-1">
                        AI Classification
                      </span>
                      <h2 className={`text-xl font-black ${activeMeta.color}`}>
                        {activeMeta.label}
                      </h2>
                      <p className="mt-1 text-xs font-medium text-slate-500">
                        Diagnostic label withheld — this image did not pass reliability checks.
                      </p>
                    </div>
                  </div>

                  {/* OOD details — ALWAYS shown when flagged */}
                  <OODDetailsCard ood={result.glaucoma.ood} />
                </>
              ) : (
                <>
                  {/* Prediction Banner */}
                  <div
                    className={`p-6 rounded-3xl border shadow-sm ${activeMeta.bg} ${activeMeta.border} flex items-center justify-between`}
                  >
                    <div>
                      <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-1">
                        AI Classification
                      </span>
                      <h2 className={`text-3xl font-black ${activeMeta.color}`}>
                        {result.glaucoma.prediction.class_name}
                      </h2>
                      <p className="mt-1 text-xs font-medium text-slate-500">
                        Based on global features and structural metrics
                      </p>
                    </div>
                    <div className="text-right">
                      <div className={`text-4xl font-black font-mono ${activeMeta.color}`}>
                        {result.glaucoma.prediction.confidence}%
                      </div>
                      <div className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                        Confidence
                      </div>
                    </div>
                  </div>

                  {/* Class Probabilities + Grad-CAM Summary — side by side */}
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2">

                    {/* Class Probabilities */}
                    <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
                      <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-5">
                        Class Probabilities
                      </span>
                      <div className="space-y-5">
                        {Object.entries(result.glaucoma.prediction.probabilities).map(([key, val]) => {
                          const meta = RISK_META[key] || RISK_META.Normal;
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
                    </div>

                    {/* Grad-CAM Explainability Summary */}
                    <GradCAMSummaryCard gradcam={result.glaucoma.gradcam} />
                  </div>

                  {/* Structural Metrics strip */}
                  <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
                    <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-4">
                      Structural Metrics
                    </span>
                    <div className="grid grid-cols-4 gap-3">
                      {[
                        { label: "CDR", val: result.cdr.cdr.value?.toFixed(2) ?? "—", color: "text-red-500" },
                        { label: "Risk Level", val: result.glaucoma.risk?.level ?? "High", color: "text-orange-500" },
                        { label: "Confidence", val: `${result.glaucoma.prediction.confidence}%`, color: "text-blue-500" },
                        { label: "Model", val: result.glaucoma.model ?? "EFF-B4", color: "text-purple-500" },
                      ].map((m, i) => (
                        <div key={i} className="p-3 text-center bg-slate-50 rounded-2xl">
                          <div className={`font-mono text-lg font-black ${m.color}`}>{m.val}</div>
                          <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mt-0.5">
                            {m.label}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}

              {/* Recommendation — always shown; backend already supplies OOD-specific text */}
              <div className="relative p-6 overflow-hidden text-white bg-indigo-900 shadow-lg rounded-3xl">
                <div className="absolute top-0 right-0 p-4 pointer-events-none opacity-10">
                  <EyeIcon size={80} />
                </div>
                <div className="relative z-10">
                  <span className="text-[10px] font-bold text-indigo-300 uppercase tracking-widest mb-2 block">
                    {isOOD ? "Recommended Next Step" : "Clinical Recommendation"}
                  </span>
                  <p className="text-sm font-medium leading-relaxed text-indigo-50">
                    {result.glaucoma.risk.recommendation}
                  </p>
                  <div className="flex gap-3 mt-4">
                    <button
                      onClick={handleReset}
                      className="bg-indigo-500 hover:bg-indigo-400 px-4 py-2 rounded-xl text-[11px] font-bold transition-all"
                    >
                      New Analysis
                    </button>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}
      </main>

      {/* Error Toast */}
      {error && (
        <div className="fixed px-4 py-2 text-xs font-bold text-red-600 -translate-x-1/2 border border-red-200 rounded-lg shadow-xl bottom-6 left-1/2 bg-red-50">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}