/**
 * OCTFundusAnalyser.jsx
 * EYE OCT Retinal Disease Analysis System — Pure Tailwind CSS, zero custom styles
 */

import { useState, useRef, useCallback } from "react";
import { predictOCT } from "../services/api_oct.js";

// ── Class metadata details ──────────────────────────────────────────────────────────
const CLASS_META = {
  CNV: {
    color: "text-red-500",
    bg: "bg-red-50",
    border: "border-red-200",
    dot: "bg-red-500",
    bar: "bg-red-500",
    accent: "border-l-red-500",
    badge: "bg-red-50 text-red-500 border-red-200",
    label: "Choroidal Neovascularization",
  },
  DME: {
    color: "text-orange-500",
    bg: "bg-orange-50",
    border: "border-orange-200",
    dot: "bg-orange-500",
    bar: "bg-orange-500",
    accent: "border-l-orange-500",
    badge: "bg-orange-50 text-orange-500 border-orange-200",
    label: "Diabetic Macular Edema",
  },
  DRUSEN: {
    color: "text-yellow-500",
    bg: "bg-yellow-50",
    border: "border-yellow-200",
    dot: "bg-yellow-500",
    bar: "bg-yellow-500",
    accent: "border-l-yellow-500",
    badge: "bg-yellow-50 text-yellow-500 border-yellow-200",
    label: "Drusen (Dry AMD)",
  },
  NORMAL: {
    color: "text-emerald-500",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
    dot: "bg-emerald-500",
    bar: "bg-emerald-500",
    accent: "border-l-emerald-500",
    badge: "bg-emerald-50 text-emerald-500 border-emerald-200",
    label: "Normal Retina",
  },
};

const CLASS_ORDER = ["CNV", "DME", "DRUSEN", "NORMAL"];

// ── Icons details OCT  Details───────────────────────────────────────────────────────────────────
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

function UploadIcon() {
  return (
    <svg
      width={22}
      height={22}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" y1="3" x2="12" y2="15" />
    </svg>
  );
}

// ── ProbabilityBar ──────────────────────────────────────────────────────────
function ProbabilityBar({ cls, value, isTop }) {
  const meta = CLASS_META[cls];
  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 ${meta.dot}`} />
          <span
            className={`text-[11px] font-mono ${isTop ? "font-bold text-slate-900" : "font-normal text-slate-400"}`}
          >
            {cls}
          </span>
          {isTop && (
            <span
              className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${meta.badge}`}
            >
              TOP
            </span>
          )}
        </div>
        <span
          className={`text-xs font-mono font-semibold ${isTop ? meta.color : "text-slate-400"}`}
        >
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-1000 ease-out ${meta.bar}`}
          style={{ width: `${value}%` }}
        />
      </div>
    </div>
  );
}

// ── ImageViewer ─────────────────────────────────────────────────────────────
function ImageViewer({ result, activeTab, setActiveTab }) {
  const TABS = [
    { key: "original", label: "Original", badge: "CLAHE Enhanced" },
    { key: "heatmap", label: "Grad-CAM", badge: "JET colormap" },
    { key: "overlay", label: "Overlay", badge: "α = 0.45" },
  ];
  const active = TABS.find((t) => t.key === activeTab);

  return (
    <div className="bg-white border border-slate-200 rounded-2xl p-3.5">
      {/* Tab strip */}
      <div className="flex gap-1 p-1 mb-3 border bg-slate-50 border-slate-200 rounded-xl">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`flex-1 py-1.5 rounded-lg text-[11px] font-mono cursor-pointer transition-all border-0
              ${
                activeTab === key
                  ? "bg-white text-slate-900 font-bold shadow-sm"
                  : "bg-transparent text-slate-400 font-normal"
              }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Main image */}
      <div className="flex items-center justify-center overflow-hidden border rounded-xl border-slate-200 bg-slate-50 aspect-square">
        <img
          src={`data:image/png;base64,${result.images[activeTab]}`}
          alt={active?.label}
          className="object-cover w-full h-full"
        />
      </div>

      {/* Badge row */}
      <div className="flex items-center justify-between mt-2">
        <span className="text-[10px] font-mono text-slate-400">
          {active?.label} View
        </span>
        <span className="text-[9px] font-mono bg-indigo-50 text-indigo-500 border border-indigo-200 px-2 py-0.5 rounded">
          {active?.badge}
        </span>
      </div>

      {/* Thumbnails */}
      <div className="grid grid-cols-3 gap-1.5 mt-2.5">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setActiveTab(key)}
            className={`rounded-lg overflow-hidden cursor-pointer p-0 bg-transparent transition-all border
              ${activeTab === key ? "border-indigo-500" : "border-slate-200"}`}
          >
            <img
              src={`data:image/png;base64,${result.images[key]}`}
              alt={label}
              className="block object-cover w-full aspect-square"
            />
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Main Component ──────────────────────────────────────────────────────────
export default function OCTFundusAnalyser() {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState("original");
  const inputRef = useRef(null);
  const resultRef = useRef(null);

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPEG, PNG, etc.)");
      return;
    }
    setFile(f);
    setError(null);
    setResult(null);
    setPreview(URL.createObjectURL(f));
  }, []);

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      handleFile(e.dataTransfer.files[0]);
    },
    [handleFile],
  );

  const handleAnalyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predictOCT(file);
      setResult(data);
      setActiveTab("original");
      setTimeout(
        () =>
          resultRef.current?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          }),
        100,
      );
    } catch (err) {
      setError(err.response?.data?.error || err.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const topClass = result
    ? Object.entries(result.class_probabilities).sort(
        (a, b) => b[1] - a[1],
      )[0][0]
    : null;

  return (
    <div className="min-h-screen w-full bg-[#f6f7f9] [background-image:radial-gradient(#d1d5db_1px,transparent_1px)] [background-size:22px_22px]">


      {/* ── Content ── */}
      <div className="max-w-[1400px] mx-auto px-8 py-8 pb-16">
        {/* ── Hero ── */}
        <div className="mb-6 text-center">
          <div className="inline-flex items-center gap-1.5 font-mono text-[10px] text-indigo-500 bg-indigo-50 border border-indigo-200 px-3 py-1 rounded-full mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
            AI-Powered Retinal OCT Analysis
          </div>
          <h1 className="text-[clamp(22px,3.5vw,32px)] font-semibold text-slate-900 tracking-tight leading-tight mb-3">
            Upload an OCT Scan for{" "}
            <span className="text-transparent bg-gradient-to-r from-indigo-500 to-violet-500 bg-clip-text">
              Instant Disease Detection
            </span>
          </h1>
          <p className="text-[14px] text-slate-500 max-w-2xl mx-auto leading-relaxed">
            Classifies <strong className="text-slate-700">CNV</strong>,{" "}
            <strong className="text-slate-700">DME</strong>,{" "}
            <strong className="text-slate-700">Drusen</strong>, and{" "}
            <strong className="text-slate-700">Normal</strong> retina from OCT
            scans using EfficientNet-B0 with Grad-CAM explainability and CLAHE
            preprocessing.
          </p>
        </div>

        {/* ── Upload zone ── */}
        <div className="max-w-2xl mx-auto mb-8">
          <div
            className={`border-2 border-dashed rounded-2xl bg-white cursor-pointer transition-all
              ${isDragging ? "border-indigo-400 bg-indigo-50/30" : "border-slate-300 hover:border-indigo-400 hover:bg-slate-50"}`}
            style={{
              padding: preview ? 14 : 32,
              textAlign: "center",
              position: "relative",
              overflow: "hidden",
            }}
            onDrop={onDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onClick={() => !loading && inputRef.current?.click()}
          >
            {/* Loading overlay */}
            {loading && (
              <div className="absolute inset-0 z-10 bg-white/90 backdrop-blur-sm rounded-2xl flex flex-col items-center justify-center gap-2.5">
                <div className="border-2 border-indigo-100 rounded-full w-11 h-11 border-t-indigo-500 animate-spin" />
                <div className="text-center">
                  <p className="text-[13px] font-semibold text-slate-700 mb-1">
                    Analysing OCT scan…
                  </p>
                  <p className="text-[11px] font-mono text-slate-400">
                    Running EfficientNet-B0 + Grad-CAM
                  </p>
                </div>
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => handleFile(e.target.files[0])}
            />

            {preview ? (
              <div>
                <div className="relative w-48 h-48 mx-auto mb-3 overflow-hidden border rounded-xl border-slate-200">
                  <img
                    src={preview}
                    alt="OCT scan"
                    className="object-cover w-full h-full"
                  />
                  {loading && (
                    <div className="absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-indigo-500 to-transparent animate-[scanLine_1.5s_ease-in-out_infinite]" />
                  )}
                </div>
                <p className="text-[12px] font-mono text-slate-500 mb-3 truncate">
                  {file?.name}
                </p>
                <div className="flex justify-center gap-2">
                  <button
                    className="inline-flex items-center gap-1.5 h-9 px-3.5 rounded-lg border border-slate-200 bg-white text-slate-500 text-xs font-medium cursor-pointer hover:border-slate-300 hover:text-slate-700 transition-all"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                  >
                    ✕ Change
                  </button>
                  <button
                    disabled={loading}
                    className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg border-0 bg-gradient-to-br from-indigo-500 to-violet-500 text-white text-[13px] font-semibold cursor-pointer shadow-lg shadow-indigo-200 hover:shadow-xl hover:shadow-indigo-300 hover:-translate-y-px transition-all disabled:bg-slate-100 disabled:text-slate-300 disabled:shadow-none disabled:cursor-not-allowed"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleAnalyse();
                    }}
                  >
                    {loading ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Analysing…
                      </>
                    ) : (
                      <>
                        <EyeIcon size={13} />
                        Analyse Scan
                      </>
                    )}
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <div
                  className={`w-12 h-12 rounded-xl mx-auto mb-3 flex items-center justify-center border transition-all
                  ${isDragging ? "bg-indigo-50 border-indigo-200" : "bg-slate-50 border-slate-200"}`}
                >
                  <UploadIcon />
                </div>
                <p
                  className={`text-[13px] font-semibold mb-1.5 ${isDragging ? "text-indigo-500" : "text-slate-700"}`}
                >
                  {isDragging
                    ? "Release to upload"
                    : "Drop your OCT image here"}
                </p>
                <p className="text-xs text-slate-400 mb-3.5">
                  or click to browse — JPEG, PNG supported
                </p>
                <span className="font-mono text-[10px] text-slate-300 bg-slate-50 border border-slate-200 px-2.5 py-1 rounded">
                  Recommended: 224 × 224 px or larger
                </span>
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="mt-2.5 px-3.5 py-2.5 rounded-xl bg-red-50 border border-red-200 text-red-700 text-[13px] flex gap-2">
              <span className="flex-shrink-0">⚠️</span>
              {error}
            </div>
          )}
        </div>

        {/* ── Class reference cards (empty state) ── */}
        {!result && !loading && (
          <div>
            <p className="text-center font-mono text-[10px] text-slate-400 tracking-widest uppercase mb-3">
              Detectable OCT Conditions
            </p>
            <div className="grid max-w-4xl grid-cols-4 gap-4 mx-auto">
              {CLASS_ORDER.map((cls) => {
                const meta = CLASS_META[cls];
                return (
                  <div
                    key={cls}
                    className={`px-5 py-5 rounded-2xl border text-center ${meta.bg} ${meta.border}`}
                  >
                    <div
                      className={`w-2.5 h-2.5 rounded-full mx-auto mb-3 ${meta.dot}`}
                    />
                    <div
                      className={`font-mono text-base font-bold mb-1.5 ${meta.color}`}
                    >
                      {cls}
                    </div>
                    <div className="text-[12px] text-slate-500 leading-snug">
                      {meta.label}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div
            ref={resultRef}
            className="flex flex-col gap-3.5 animate-[fadeUp_0.5s_cubic-bezier(0.22,1,0.36,1)_both]"
          >
            {/* Prediction banner */}
            {(() => {
              const meta = CLASS_META[result.prediction] || CLASS_META.NORMAL;
              return (
                <div
                  className={`px-5 py-4 rounded-2xl border flex items-center justify-between flex-wrap gap-3 ${meta.bg} ${meta.border}`}
                >
                  <div className="flex items-center gap-3.5">
                    <div
                      className={`w-11 h-11 rounded-xl bg-white border flex items-center justify-center ${meta.border}`}
                    >
                      <span className={`text-xl font-mono font-bold ${meta.color}`}>
                        {result.prediction[0]}
                      </span>
                    </div>
                    <div>
                      <span className="font-mono text-[9px] text-slate-400 tracking-widest uppercase block mb-1">
                        Predicted Diagnosis
                      </span>
                      <h2 className="text-xl font-bold tracking-tight text-slate-900">
                        {meta.label}
                      </h2>
                      <span className="font-mono text-[10px] text-slate-400">
                        OCT Code: {result.prediction}
                      </span>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="font-mono text-[10px] text-slate-400 block mb-1">
                      Model Confidence
                    </span>
                    <span className={`font-mono text-[46px] font-bold leading-none ${meta.color}`}>
                      {result.confidence.toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })()}
          </div>
        )}
      </div>
    </div>
  );
}