import { useState, useRef, useCallback } from "react";
import { analyzeRetina } from "../services/api_dr";

/* ─── Grade meta ─────────────────────────────────────────────────────────── */
const GRADE_META = [
  {
    label: "No DR",
    color: "#10b981",
    bg: "bg-emerald-50",
    border: "border-emerald-200",
  },
  {
    label: "Mild DR",
    color: "#84cc16",
    bg: "bg-lime-50",
    border: "border-lime-200",
  },
  {
    label: "Moderate DR",
    color: "#f59e0b",
    bg: "bg-amber-50",
    border: "border-amber-200",
  },
  {
    label: "Severe DR",
    color: "#ea580c",
    bg: "bg-orange-50",
    border: "border-orange-200",
  },
  {
    label: "Proliferative DR",
    color: "#ef4444",
    bg: "bg-red-50",
    border: "border-red-200",
  },
];

/* ─── Icons ─────────────────────────────────────────────────────────────── */
function EyeIcon({ size = 24, color = "#6366f1" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

/* ─── Main Component ─────────────────────────────────────────────────────── */
export default function DiabeticRetinopathy() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

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
      const data = await analyzeRetina(file);
      setResult(data);
    } catch (e) {
      setError(e.response?.data?.error || e.message || "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  const activeMeta = result ? GRADE_META[result.grade] || GRADE_META[0] : null;

  return (
    <div
      className="w-full min-h-screen text-slate-900"
      style={dotBackgroundStyle}
    >


      <main className="max-w-2xl px-6 py-12 mx-auto">
        <div className="mb-10 text-center">
          <h1 className="text-2xl font-bold text-slate-900">
            DR Classification
          </h1>
          <p className="mt-1 text-xs text-slate-500">
            Single-scan fundus diagnostic output
          </p>
        </div>

        {/* Action Card */}
        <div className="relative p-8 overflow-hidden bg-white border shadow-sm rounded-3xl border-slate-200">
          {/* Loading Overlay */}
          {loading && (
            <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/90 backdrop-blur-sm animate-in fade-in">
              <div className="w-10 h-10 mb-3 border-4 border-indigo-100 rounded-full border-t-indigo-600 animate-spin"></div>
              <p className="text-[10px] font-bold text-indigo-600 uppercase tracking-widest">
                Processing Fundus...
              </p>
            </div>
          )}

          <div className="flex flex-col items-center">
            {preview ? (
              <div className="flex flex-col items-center w-full space-y-6">
                <img
                  src={preview}
                  alt="Preview"
                  className="object-cover w-48 h-48 border shadow-sm rounded-2xl border-slate-100"
                />

                {!result && !loading && (
                  <button
                    onClick={handleAnalyse}
                    className="px-10 py-3 text-xs font-bold text-white transition-all bg-indigo-600 shadow-lg hover:bg-indigo-700 rounded-xl shadow-indigo-100"
                  >
                    RUN CLASSIFICATION
                  </button>
                )}

                {/* Simplified Classification Result */}
                {result && (
                  <div
                    className={`w-full p-6 rounded-2xl border ${activeMeta.bg} ${activeMeta.border} animate-in zoom-in-95 duration-500`}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
                        Diagnostic Result
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-white/50 border border-current opacity-60">
                        Grade {result.grade}
                      </span>
                    </div>

                    <div className="flex flex-col items-center text-center">
                      <h2
                        className="text-3xl font-black tracking-tight uppercase"
                        style={{ color: activeMeta.color }}
                      >
                        {result.label}
                      </h2>
                      <div className="flex items-center gap-4 mt-4">
                        <div className="text-center">
                          <p className="text-[10px] text-slate-400 uppercase font-bold">
                            Confidence
                          </p>
                          <p className="text-xl font-black text-slate-800">
                            {result.confidence.toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="pt-6 mt-6 border-t border-slate-200/50">
                      <button
                        onClick={() => {
                          setFile(null);
                          setPreview(null);
                          setResult(null);
                        }}
                        className="w-full py-2.5 bg-white border border-slate-200 text-slate-600 rounded-xl text-xs font-bold hover:bg-slate-50 transition-all"
                      >
                        NEW ANALYSIS
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div
                onClick={() => document.getElementById("dr-upload").click()}
                className="flex flex-col items-center w-full py-16 transition-colors border-2 border-dashed cursor-pointer border-slate-200 rounded-2xl hover:border-indigo-300"
              >
                <div className="flex items-center justify-center w-12 h-12 mb-3 bg-slate-50 rounded-xl">
                  <svg
                    className="w-5 h-5 text-slate-400"
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
                <p className="text-xs font-bold tracking-wide uppercase text-slate-600">
                  Select Fundus Image
                </p>
                <input
                  id="dr-upload"
                  type="file"
                  className="hidden"
                  onChange={(e) => handleFile(e.target.files[0])}
                />
              </div>
            )}
          </div>
        </div>

        {/* Minimal Footer */}
        {error && (
          <div className="p-4 mt-6 text-xs font-bold text-center text-red-600 border border-red-100 bg-red-50 rounded-xl animate-pulse">
            ⚠️ {error}
          </div>
        )}
      </main>
    </div>
  );
}
