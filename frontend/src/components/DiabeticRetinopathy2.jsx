import { useState, useRef, useCallback } from "react";
import { analyzeRetina } from "../services/api_dr";

/* ─── Grade meta ─────────────────────────────────────────────────────────── */
const GRADE_META = [
  {
    label: "No DR",
    color: "#10b981",
    bg: "rgba(16,185,129,0.08)",
    ring: "#bbf7d0",
  },
  {
    label: "Mild DR",
    color: "#84cc16",
    bg: "rgba(132,204,22,0.08)",
    ring: "#d9f99d",
  },
  {
    label: "Moderate DR",
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.08)",
    ring: "#fef3c7",
  },
  {
    label: "Severe DR",
    color: "#ea580c",
    bg: "rgba(234,88,12,0.08)",
    ring: "#ffedd5",
  },
  {
    label: "Proliferative DR",
    color: "#ef4444",
    bg: "rgba(239,68,68,0.08)",
    ring: "#fee2e2",
  },
];

/* ─── EyeIcon ─────────────────────────────────────────────────────────────── */
function EyeIcon({ size = 24, color = "#6366f1", animated = false }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className={animated ? "animate-pulse" : ""}
    >
      <ellipse cx="12" cy="12" rx="9" ry="6" stroke={color} strokeWidth="1.5" />
      <circle cx="12" cy="12" r="3" fill={color} opacity="0.9" />
      <circle cx="13" cy="11" r="0.8" fill="white" opacity="0.7" />
    </svg>
  );
}

/* ─── Loading Overlay ────────────────────────────────────────────────────── */
function LoadingState() {
  return (
    <div className="absolute inset-0 z-50 flex flex-col items-center justify-center duration-300 bg-white/80 backdrop-blur-sm rounded-3xl animate-in fade-in">
      <div className="relative flex items-center justify-center w-20 h-20 mb-4">
        {/* Outer spinning ring */}
        <div className="absolute inset-0 border-4 border-indigo-100 rounded-full border-t-indigo-600 animate-spin" />
        {/* Inner pulsing eye */}
        <EyeIcon size={32} color="#4f46e5" animated />
      </div>
      <div className="text-center">
        <p className="text-sm font-bold tracking-wide uppercase text-slate-800 animate-pulse">
          Analysing Retina
        </p>
        <p className="text-[10px] font-mono text-slate-400 mt-1 uppercase tracking-widest">
          Detecting Lesions & Grading
        </p>
      </div>
    </div>
  );
}

/* ─── ProbabilityBar ─────────────────────────────────────────────────────── */
function ProbabilityBar({ name, value, color, isMax }) {
  return (
    <div className="mb-2.5">
      <div className="flex items-center justify-between mb-1">
        <span
          className="text-[10px] font-mono uppercase tracking-tight"
          style={{ color: isMax ? color : "#94a3b8" }}
        >
          {name}
        </span>
        <span className="text-[10px] font-mono font-bold" style={{ color }}>
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
        <div
          className="h-full transition-all duration-1000 ease-out rounded-full"
          style={{
            width: `${value}%`,
            background: isMax ? color : "#cbd5e1",
          }}
        />
      </div>
    </div>
  );
}

/* ─── LesionBadge ────────────────────────────────────────────────────────── */
function LesionBadge({ label, count, color, icon }) {
  return (
    <div className="flex items-center justify-between px-3 py-2 transition-all border rounded-lg border-slate-100 bg-slate-50/50 hover:border-slate-200">
      <div className="flex items-center gap-2">
        <span className="text-sm">{icon}</span>
        <span className="text-[11px] font-mono font-medium text-slate-500 uppercase">
          {label}
        </span>
      </div>
      <span className="font-mono text-sm font-bold text-slate-900">
        {count}
      </span>
    </div>
  );
}

/* ─── ImagePanel ─────────────────────────────────────────────────────────── */
function ImagePanel({ title, src, tag, tagColor = "#6366f1" }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest">
          {title}
        </span>
        {tag && (
          <span
            className="text-[9px] font-bold font-mono px-1.5 py-0.5 rounded border"
            style={{
              color: tagColor,
              background: `${tagColor}10`,
              borderColor: `${tagColor}30`,
            }}
          >
            {tag}
          </span>
        )}
      </div>
      <div className="relative overflow-hidden border aspect-square rounded-xl border-slate-200 bg-slate-50">
        {src ? (
          <img
            src={`data:image/png;base64,${src}`}
            alt={title}
            className="object-cover w-full h-full"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-slate-300 text-[10px] font-mono italic">
            No Data
          </div>
        )}
      </div>
    </div>
  );
}

/* ─── OODBadge ───────────────────────────────────────────────────────────── */
/* Purely about the Mahalanobis OOD check — used on the OOD detail card. */
function OODBadge({ isOod }) {
  const color = isOod ? "#ef4444" : "#10b981";
  return (
    <span
      className="text-[9px] font-bold font-mono px-2 py-1 rounded-full border flex items-center gap-1"
      style={{
        color,
        background: `${color}10`,
        borderColor: `${color}30`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: color }}
      />
      {isOod ? "OUT-OF-DISTRIBUTION" : "IN-DISTRIBUTION"}
    </span>
  );
}

/* ─── ReliabilityBadge ───────────────────────────────────────────────────── */
/* Covers both OOD and low-confidence — used on the main classification card. */
const RELIABILITY_LABELS = {
  ood: "OUT-OF-DISTRIBUTION",
  low_confidence: "LOW CONFIDENCE",
  ood_and_low_confidence: "OOD + LOW CONFIDENCE",
};

function ReliabilityBadge({ isFlagged, reason }) {
  const color = isFlagged ? "#ef4444" : "#10b981";
  const text = isFlagged ? RELIABILITY_LABELS[reason] || "FLAGGED" : "RELIABLE";
  return (
    <span
      className="text-[9px] font-bold font-mono px-2 py-1 rounded-full border flex items-center gap-1"
      style={{
        color,
        background: `${color}10`,
        borderColor: `${color}30`,
      }}
    >
      <span
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: color }}
      />
      {text}
    </span>
  );
}

/* ─── OODCard ────────────────────────────────────────────────────────────── */
function OODCard({ ood }) {
  const color = ood.is_ood ? "#ef4444" : "#10b981";
  const pct = Math.min(
    (ood.score / Math.max(ood.threshold * 1.4, ood.score * 1.05)) * 100,
    100,
  );
  const threshPct = Math.min(
    (ood.threshold / Math.max(ood.threshold * 1.4, ood.score * 1.05)) * 100,
    100,
  );

  return (
    <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest font-bold">
          OOD Detection
        </span>
        <OODBadge isOod={ood.is_ood} />
      </div>

      <div className="relative h-2 mb-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="absolute top-0 bottom-0 transition-all duration-1000 ease-out rounded-full"
          style={{ width: `${pct}%`, background: color }}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-slate-900"
          style={{ left: `${threshPct}%` }}
        />
      </div>
      <div className="flex justify-between mb-4 font-mono text-[10px] text-slate-400">
        <span>score {ood.score.toFixed(3)}</span>
        <span>threshold {ood.threshold.toFixed(3)}</span>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        <div className="px-3 py-2 border rounded-lg border-slate-100 bg-slate-50/50">
          <span className="block text-[9px] font-mono uppercase text-slate-400">
            Method
          </span>
          <span className="font-mono text-xs font-bold capitalize text-slate-900">
            {ood.method}
          </span>
        </div>
        <div className="px-3 py-2 border rounded-lg border-slate-100 bg-slate-50/50">
          <span className="block text-[9px] font-mono uppercase text-slate-400">
            Mahalanobis Score
          </span>
          <span className="font-mono text-xs font-bold" style={{ color }}>
            {ood.score.toFixed(3)}
          </span>
        </div>
      </div>
    </div>
  );
}

/* ─── ConfidenceCard ─────────────────────────────────────────────────────── */
function ConfidenceCard({ confidence, threshold, isLow }) {
  const color = isLow ? "#ef4444" : "#10b981";
  const pct = Math.min(confidence, 100);

  return (
    <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
      <div className="flex items-center justify-between mb-4">
        <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest font-bold">
          Confidence Reliability
        </span>
        <span
          className="text-[9px] font-bold font-mono px-2 py-1 rounded-full border flex items-center gap-1"
          style={{ color, background: `${color}10`, borderColor: `${color}30` }}
        >
          <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
          {isLow ? "BELOW THRESHOLD" : "ABOVE THRESHOLD"}
        </span>
      </div>

      <div className="relative h-2 mb-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className="absolute top-0 bottom-0 transition-all duration-1000 ease-out rounded-full"
          style={{ width: `${pct}%`, background: color }}
        />
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-slate-900"
          style={{ left: `${threshold}%` }}
        />
      </div>
      <div className="flex justify-between mb-4 font-mono text-[10px] text-slate-400">
        <span>confidence {confidence.toFixed(1)}%</span>
        <span>threshold {threshold.toFixed(0)}%</span>
      </div>
    </div>
  );
}

/* ─── ReliabilityWarningBanner ───────────────────────────────────────────── */
/* Shown for OOD, low confidence, or both — same visual weight either way. */
function ReliabilityWarningBanner({ reliability, ood, confidence }) {
  const { reason } = reliability;

  let title = "Unreliable Prediction";
  let body =
    "The grade and lesion counts below may be unreliable. Please verify the image and re-submit.";

  if (reason === "ood") {
    title = "Out-of-Distribution Image Detected";
    body =
      "This image's feature embedding falls outside the training distribution — it may not be a valid retinal fundus photo, or is an unusual / poor-quality capture. Detailed grading has been withheld until a valid fundus image is submitted.";
  } else if (reason === "low_confidence") {
    title = "Low-Confidence Prediction";
    body = `The model's confidence in this prediction is only ${confidence.toFixed(1)}%, below the ${reliability.confidence_threshold.toFixed(0)}% reliability threshold. The grade and lesion counts below may be unreliable.`;
  } else if (reason === "ood_and_low_confidence") {
    title = "Out-of-Distribution & Low-Confidence Prediction";
    body = `This image falls outside the training distribution. Detailed grading has been withheld until a valid fundus image is submitted.`;
  }

  return (
    <div className="p-5 border-2 border-red-200 shadow-sm bg-red-50 rounded-2xl">
      <div className="flex items-start gap-3">
        <span className="text-lg leading-none">⚠️</span>
        <div>
          <p className="text-xs font-bold tracking-wide text-red-700 uppercase">
            {title}
          </p>
          <p className="mt-1 text-xs leading-relaxed text-red-600">{body}</p>
        </div>
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function DiabeticRetinopathy() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  const dotBackgroundStyle = {
    backgroundColor: "#f6f7f9",
    backgroundImage: "radial-gradient(#d1d5db 1px, transparent 1px)",
    backgroundSize: "22px 22px",
  };

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  }, []);

  const handleAnalyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await analyzeRetina(file);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.error || e.message || "Analysis failed");
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

  // Fallback so older backend responses without `reliability` still render.
  const reliability = result?.reliability || {
    is_flagged: !!result?.ood?.is_ood,
    is_low_confidence: false,
    confidence_threshold: 70,
    reason: result?.ood?.is_ood ? "ood" : null,
  };

  // OOD specifically (not low-confidence) hides the grading/explainability
  // sections entirely, since the prediction underneath isn't even on a
  // retinal image — showing a "grade" for it would be misleading.
  const isOod = !!result?.ood?.is_ood;

  return (
    <div
      className="w-full min-h-screen font-sans text-slate-900"
      style={dotBackgroundStyle}
    >

      <main className="max-w-[1200px] mx-auto px-6 py-10">
        <div className="mb-10 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-slate-900">
            Diabetic Retinopathy Screening
          </h1>
          <p className="text-sm text-slate-500">
            Expert-level DR grading with clinical lesion localization
          </p>
        </div>

        <div className="grid items-start grid-cols-1 gap-8 lg:grid-cols-12">
          {/* LEFT: Upload & Result Summary */}
          <div className="space-y-6 lg:col-span-5">
            <div
              className={`relative rounded-3xl border-2 border-dashed p-8 transition-all bg-white shadow-sm overflow-hidden
                ${isDragging ? "border-indigo-400 bg-indigo-50/50" : "border-slate-200 hover:border-indigo-300"}`}
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                handleFile(e.dataTransfer.files[0]);
              }}
            >
              {/* Loading Spinner Overlay */}
              {loading && <LoadingState />}

              <div className="flex flex-col items-center">
                {preview ? (
                  <div className="duration-500 animate-in fade-in zoom-in-95">
                    <img
                      src={preview}
                      alt="Fundus"
                      className="object-cover w-64 h-64 mb-4 border shadow-sm rounded-2xl border-slate-100"
                    />
                    {!loading && (
                      <div className="flex justify-center gap-2">
                        <button
                          onClick={handleAnalyse}
                          className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2.5 rounded-xl text-xs font-bold shadow-lg shadow-indigo-200 transition-all"
                        >
                          START AI SCAN
                        </button>
                        <button
                          onClick={handleReset}
                          className="bg-slate-100 hover:bg-slate-200 text-slate-600 px-4 py-2.5 rounded-xl text-xs font-bold transition-all"
                        >
                          CHANGE
                        </button>
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    className="flex flex-col items-center w-full py-10 cursor-pointer"
                    onClick={() =>
                      document.getElementById("retina-upload-input").click()
                    }
                  >
                    <div className="flex items-center justify-center w-16 h-16 mb-4 border bg-slate-50 rounded-2xl border-slate-100">
                      <svg
                        className="w-6 h-6 text-slate-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="M12 4v16m8-8H4"
                        />
                      </svg>
                    </div>
                    <p className="text-sm font-bold tracking-wide uppercase text-slate-700">
                      Upload Fundus Photo
                    </p>
                    <p className="mt-1 font-mono text-xs italic text-slate-400">
                      Drop image or click here
                    </p>
                    <input
                      id="retina-upload-input"
                      type="file"
                      className="hidden"
                      onChange={(e) => handleFile(e.target.files[0])}
                    />
                  </div>
                )}
              </div>
            </div>

            {/* AI Result Side Cards */}
            {result && !loading && (
              <div className="space-y-6 duration-500 animate-in slide-in-from-bottom-4">
                {/* Warning Banner — shown for OOD, low confidence, or both */}
                {reliability.is_flagged && (
                  <ReliabilityWarningBanner
                    reliability={reliability}
                    ood={result.ood}
                    confidence={result.confidence}
                  />
                )}

                {/* Severity Card — hidden on OOD, grade isn't meaningful */}
                {!isOod &&
                  (() => {
                    const meta = GRADE_META[result.grade] || GRADE_META[0];
                    return (
                      <div
                        className="p-6 bg-white border border-l-4 shadow-sm border-slate-200 rounded-3xl"
                        style={{ borderLeftColor: meta.color }}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest">
                            AI Classification
                          </span>
                          <ReliabilityBadge
                            isFlagged={reliability.is_flagged}
                            reason={reliability.reason}
                          />
                        </div>
                        <h2
                          className="text-2xl font-black tracking-tight uppercase"
                          style={{ color: meta.color }}
                        >
                          {result.label}
                        </h2>
                        <div className="flex items-end justify-between pt-4 mt-4 border-t border-slate-50">
                          <span className="font-mono text-xs tracking-tighter uppercase text-slate-400">
                            Confidence Score
                          </span>
                          <span
                            className="text-lg font-black"
                            style={{
                              color: reliability.is_low_confidence
                                ? "#ef4444"
                                : "#0f172a",
                            }}
                          >
                            {result.confidence.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    );
                  })()}

                {/* Lesion Counter — hidden on OOD */}
                {!isOod && (
                  <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
                    <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-4">
                      Clinical Findings
                    </span>
                    <div className="grid grid-cols-1 gap-2.5">
                      <LesionBadge
                        label="Microaneurysms"
                        count={result.lesions.microaneurysms}
                        color="#f59e0b"
                        icon="🔴"
                      />
                      <LesionBadge
                        label="Hemorrhages"
                        count={result.lesions.hemorrhages}
                        color="#ef4444"
                        icon="🩸"
                      />
                      <LesionBadge
                        label="Hard Exudates"
                        count={result.lesions.hard_exudates}
                        color="#6366f1"
                        icon="✨"
                      />
                    </div>
                  </div>
                )}

                {/* Confidence Detail Card — hidden on OOD */}
                {!isOod && (
                  <ConfidenceCard
                    confidence={result.confidence}
                    threshold={reliability.confidence_threshold}
                    isLow={reliability.is_low_confidence}
                  />
                )}

                {/* OOD Detection Detail Card — always shown when present */}
                {result.ood && <OODCard ood={result.ood} />}
              </div>
            )}
          </div>

          {/* RIGHT: Detailed Clinical Dashboard */}
          <div className="lg:col-span-7">
            {!result || loading ? (
              <div className="flex flex-col items-center justify-center h-full p-12 border-2 border-dashed border-slate-200 rounded-3xl opacity-40 bg-white/50">
                <EyeIcon size={48} color="#cbd5e1" />
                <p className="mt-4 font-mono text-xs tracking-widest uppercase text-slate-400">
                  Awaiting Analysis Data
                </p>
              </div>
            ) : (
              <div className="space-y-6 duration-700 animate-in fade-in slide-in-from-right-4">
                {/* Explainability Section — hidden on OOD */}
                {!isOod && (
                  <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
                    <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-6 font-bold">
                      Pathology Localization (Grad-CAM)
                    </span>
                    <div className="grid grid-cols-3 gap-4">
                      <ImagePanel
                        title="Input"
                        src={result.original_b64}
                        tag="RGB"
                      />
                      <ImagePanel
                        title="Activation"
                        src={result.gradcam_b64}
                        tag="HEATMAP"
                        tagColor="#f59e0b"
                      />
                      <ImagePanel
                        title="Fused"
                        src={result.overlay_b64}
                        tag="MAPPED"
                        tagColor="#10b981"
                      />
                    </div>
                  </div>
                )}

                {/* Probability Distribution — hidden on OOD */}
                {!isOod && (
                  <div className="p-6 bg-white border shadow-sm border-slate-200 rounded-3xl">
                    <span className="font-mono text-[10px] text-slate-400 uppercase tracking-widest block mb-6 font-bold">
                      Inference Probability Distribution
                    </span>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-1">
                      {Object.entries(result.probabilities).map(
                        ([name, val], i) => (
                          <ProbabilityBar
                            key={name}
                            name={name}
                            value={val}
                            color={GRADE_META[i]?.color || "#6366f1"}
                            isMax={i === result.grade}
                          />
                        ),
                      )}
                    </div>
                  </div>
                )}

                {/* Final Recommendation — always shown, tells the user to re-submit on OOD */}
                <div className="relative p-8 overflow-hidden text-white bg-indigo-900 shadow-xl rounded-3xl">
                  <div className="absolute top-0 right-0 p-4 opacity-10">
                    <EyeIcon size={120} color="#fff" />
                  </div>
                  <div className="relative z-10 space-y-6">
                    <div>
                      <span className="text-[10px] font-bold text-indigo-300 uppercase tracking-widest mb-2 block">
                        AI Interpretation
                      </span>
                      <p className="text-sm font-medium leading-relaxed text-indigo-50">
                        {result.explanation}
                      </p>
                    </div>
                    <div className="p-4 border bg-white/10 backdrop-blur-md border-white/10 rounded-2xl">
                      <span className="text-[10px] font-bold text-amber-400 uppercase tracking-widest mb-2 block">
                        Medical Action Plan
                      </span>
                      <p className="text-xs italic font-medium leading-relaxed text-slate-200">
                        "{result.recommendation}"
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Fixed Error Toast */}
      {error && (
        <div className="fixed px-6 py-3 text-xs font-bold text-red-600 -translate-x-1/2 bg-white border border-red-200 rounded-full shadow-2xl bottom-6 left-1/2 animate-in slide-in-from-bottom-8">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
}