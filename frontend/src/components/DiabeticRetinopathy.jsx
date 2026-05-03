import { useState, useRef, useCallback } from "react";
import { analyzeRetina } from "../services/api_dr";

/* ─── Grade meta ─────────────────────────────────────────────────────────── */
const GRADE_META = [
  {
    label: "No DR",
    color: "#059669",
    bg: "bg-emerald-50",
    border: "border-emerald-300",
    badge: "bg-emerald-100",
    badgeText: "text-emerald-700",
  },
  {
    label: "Mild DR",
    color: "#65a30d",
    bg: "bg-lime-50",
    border: "border-lime-300",
    badge: "bg-lime-100",
    badgeText: "text-lime-700",
  },
  {
    label: "Moderate DR",
    color: "#d97706",
    bg: "bg-amber-50",
    border: "border-amber-300",
    badge: "bg-amber-100",
    badgeText: "text-amber-700",
  },
  {
    label: "Severe DR",
    color: "#dc2626",
    bg: "bg-orange-50",
    border: "border-orange-300",
    badge: "bg-orange-100",
    badgeText: "text-orange-700",
  },
  {
    label: "Proliferative DR",
    color: "#991b1b",
    bg: "bg-red-50",
    border: "border-red-300",
    badge: "bg-red-100",
    badgeText: "text-red-700",
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

  const clinicalBackgroundStyle = {
    backgroundColor: "#f0f5fa",
    backgroundImage: `
      linear-gradient(135deg, rgba(15, 23, 42, 0.02) 25%, transparent 25%),
      linear-gradient(225deg, rgba(15, 23, 42, 0.02) 25%, transparent 25%),
      linear-gradient(45deg, rgba(15, 23, 42, 0.02) 25%, transparent 25%),
      linear-gradient(315deg, rgba(15, 23, 42, 0.02) 25%, transparent 25%)
    `,
    backgroundSize: "40px 40px",
    backgroundPosition: "0 0, 10px 0, 10px -10px, 0px -10px",
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
    <div className="w-full min-h-screen bg-slate-50">
      <main className="max-w-5xl px-6 py-8 mx-auto">
        {/* Minimal Header */}
        <div className="mb-8 border-b border-slate-200 pb-6">
          <h1 className="text-3xl font-bold text-slate-900">Diabetic Retinopathy</h1>
          <p className="text-sm text-slate-600 mt-1">AI-Assisted Classification System</p>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left: Image Upload */}
          <div className="space-y-6">
            {/* Upload Card */}
            <div className="relative overflow-hidden bg-white rounded-2xl shadow border border-slate-200">
              {/* Loading Overlay */}
              {loading && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/95">
                  <div className="w-12 h-12 mb-3 border-3 border-slate-200 border-t-slate-900 rounded-full animate-spin"></div>
                  <p className="text-sm font-semibold text-slate-900">Analyzing...</p>
                </div>
              )}

              {preview ? (
                <div className="p-6 space-y-4">
                  <div className="relative bg-slate-900 rounded-xl overflow-hidden">
                    <img
                      src={preview}
                      alt="Fundus"
                      className="w-full h-auto object-cover aspect-square"
                    />
                  </div>
                  <div className="text-center">
                    <p className="text-xs text-slate-500 font-semibold mb-3 uppercase tracking-wide">
                      Image Ready for Analysis
                    </p>
                    {!result && !loading && (
                      <button
                        onClick={handleAnalyse}
                        className="w-full px-6 py-3 bg-slate-900 text-white font-bold text-sm rounded-lg hover:bg-slate-800 transition"
                      >
                        ANALYZE
                      </button>
                    )}
                  </div>
                </div>
              ) : (
                <div
                  onClick={() => document.getElementById("dr-upload").click()}
                  className="p-8 text-center cursor-pointer hover:bg-slate-50 transition"
                >
                  <svg
                    className="w-10 h-10 text-slate-400 mx-auto mb-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="1.5"
                      d="M12 4v16m8-8H4"
                    />
                  </svg>
                  <p className="font-bold text-slate-900 mb-1">Upload Image</p>
                  <p className="text-xs text-slate-500">JPG, PNG • Max 10MB</p>
                  <input
                    id="dr-upload"
                    type="file"
                    className="hidden"
                    onChange={(e) => handleFile(e.target.files[0])}
                  />
                </div>
              )}
            </div>

            {/* Error */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl">
                <p className="text-sm font-bold text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div className="space-y-6">
            {result && (
              <div className="animate-in fade-in zoom-in-95 duration-500 space-y-6">
                {/* Main Result Card */}
                <div
                  className={`p-8 rounded-2xl border-2 ${activeMeta.bg} ${activeMeta.border}`}
                >
                  <div className="mb-6">
                    <p className="text-xs font-bold text-slate-600 uppercase tracking-widest mb-2">
                      Classification
                    </p>
                    <h2
                      className="text-4xl font-black"
                      style={{ color: activeMeta.color }}
                    >
                      {result.label}
                    </h2>
                  </div>

                  <p className="text-sm text-slate-700 mb-6">
                    {result.label === "No DR" && "No diabetic retinopathy detected. Continue regular monitoring."}
                    {result.label === "Mild DR" && "Early signs detected. Recommend routine follow-up and glycemic control."}
                    {result.label === "Moderate DR" && "Moderate changes present. Close monitoring and specialist referral advised."}
                    {result.label === "Severe DR" && "Severe findings detected. Urgent specialist evaluation recommended."}
                    {result.label === "Proliferative DR" && "Advanced stage detected. Immediate specialist intervention required."}
                  </p>

                  {/* Grade Badge */}
                  <div className="flex items-center gap-3 mb-6 pb-6 border-b border-current border-opacity-20">
                    <span className={`px-3 py-1.5 rounded-full ${activeMeta.badge} ${activeMeta.badgeText} text-xs font-bold`}>
                      Grade {result.grade}
                    </span>
                    <span className="text-xs text-slate-600 font-semibold">SEVERITY LEVEL</span>
                  </div>

                  {/* Confidence */}
                  <div className="mb-4">
                    <div className="flex justify-between items-baseline mb-2">
                      <p className="text-xs font-bold text-slate-600 uppercase tracking-widest">Model Confidence</p>
                      <p className="text-2xl font-black text-slate-900">{result.confidence.toFixed(1)}%</p>
                    </div>
                    <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${result.confidence}%`,
                          backgroundColor: activeMeta.color,
                        }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Grade Reference */}
                <div className="bg-white p-6 rounded-xl border border-slate-200">
                  <p className="text-xs font-bold text-slate-700 uppercase tracking-widest mb-4">
                    DR Severity Scale
                  </p>
                  <div className="grid grid-cols-5 gap-2">
                    {GRADE_META.map((grade, idx) => (
                      <div key={idx} className="text-center">
                        <div
                          className={`w-full p-2 rounded-lg text-white font-bold text-sm mb-1 transition-all ${
                            result.grade === idx ? "ring-2 ring-slate-900 ring-offset-2" : "opacity-60"
                          }`}
                          style={{ backgroundColor: grade.color }}
                        >
                          {idx}
                        </div>
                        <p className="text-xs text-slate-600 font-semibold leading-tight">
                          {grade.label.replace(" DR", "")}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Action Button */}
                <button
                  onClick={() => {
                    setFile(null);
                    setPreview(null);
                    setResult(null);
                  }}
                  className="w-full py-3 bg-white border border-slate-300 text-slate-700 font-bold rounded-lg hover:bg-slate-50 transition"
                >
                  New Analysis
                </button>
              </div>
            )}

            {!result && (
              <div className="bg-white p-8 rounded-2xl border border-slate-200 text-center text-slate-500 min-h-96 flex flex-col items-center justify-center">
                <svg className="w-12 h-12 mx-auto mb-3 opacity-20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                <p className="font-semibold">Upload a fundus image to begin analysis</p>
              </div>
            )}
          </div>
        </div>

        {/* Clinical Notes Footer */}
        <div className="mt-12 p-6 bg-blue-50 border border-blue-200 rounded-xl">
          <p className="text-xs font-bold text-blue-900 uppercase tracking-widest mb-2">⚠ Clinical Use Notice</p>
          <p className="text-sm text-blue-900">
            This AI-assisted tool is intended to support clinical decision-making only. Results must be reviewed by a qualified ophthalmologist. Do not use as the sole basis for diagnosis or treatment decisions.
          </p>
        </div>
      </main>
    </div>
  );
}
