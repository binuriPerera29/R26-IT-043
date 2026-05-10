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
    <div className="w-full min-h-screen bg-gradient-to-br from-white via-blue-50 to-blue-100">
      <main className="max-w-6xl px-6 py-12 mx-auto">
        {/* Professional Header */}
        <div className="mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-100 rounded-full mb-4 border border-blue-200">
            <div className="w-2.5 h-2.5 bg-blue-600 rounded-full animate-pulse"></div>
            <span className="text-xs font-bold text-blue-900 uppercase tracking-wider">AI-Powered Diagnostic Tool</span>
          </div>
          <h1 className="text-5xl font-bold text-gray-900 mb-2">Diabetic Retinopathy</h1>
          <p className="text-lg text-gray-600">Advanced AI-Assisted Classification System</p>
        </div>

        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
          {/* Left: Image Upload */}
          <div className="space-y-6">
            {/* Upload Card */}
            <div className="relative overflow-hidden bg-white rounded-3xl shadow-lg border-2 border-blue-200 transition-all hover:shadow-xl">
              {/* Loading Overlay */}
              {loading && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-white/95 backdrop-blur-sm">
                  <div className="relative w-16 h-16 mb-4">
                    <div className="absolute inset-0 rounded-full border-4 border-blue-200"></div>
                    <div className="absolute inset-0 rounded-full border-4 border-blue-600 border-t-transparent animate-spin"></div>
                  </div>
                  <p className="text-base font-semibold text-gray-900">Processing</p>
                  <p className="text-sm text-gray-500 mt-1">Analyzing fundus image</p>
                </div>
              )}

              {preview ? (
                <div className="p-8 space-y-6">
                  <div className="flex items-center gap-2 text-sm font-semibold text-blue-700 bg-blue-50 px-3 py-2 rounded-lg w-fit">
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Image Ready
                  </div>
                  <div className="relative bg-gradient-to-br from-gray-100 to-gray-200 rounded-2xl overflow-hidden border-2 border-blue-200 shadow-md">
                    <img
                      src={preview}
                      alt="Fundus Image"
                      className="w-full h-auto object-cover aspect-square"
                    />
                  </div>
                  <div className="bg-blue-50 rounded-xl p-4 border border-blue-200 space-y-2">
                    <p className="text-xs font-bold text-blue-900 uppercase tracking-wider">Details</p>
                    <div className="text-xs text-gray-700 space-y-1">
                      <p>Size: <span className="font-semibold">{(file?.size / (1024 * 1024)).toFixed(2)} MB</span></p>
                    </div>
                  </div>
                  {!result && !loading && (
                    <button
                      onClick={handleAnalyse}
                      className="w-full px-6 py-4 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-bold rounded-xl transition-all duration-300 shadow-lg hover:shadow-blue-300/50"
                    >
                      Analyze Image
                    </button>
                  )}
                </div>
              ) : (
                <div
                  onClick={() => document.getElementById("dr-upload").click()}
                  className="p-12 text-center cursor-pointer group transition-all hover:bg-blue-50"
                >
                  <div className="flex flex-col items-center">
                    <div className="w-24 h-24 mb-6 rounded-2xl bg-gradient-to-br from-blue-100 to-blue-50 flex items-center justify-center border-2 border-blue-200 group-hover:border-blue-400 transition-colors shadow-md">
                      <svg className="w-12 h-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-2xl font-bold text-gray-900 mb-2">Upload Fundus Image</p>
                    <p className="text-gray-600 mb-6">JPG, PNG • Max 10MB</p>
                    <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-xl transition-colors shadow-md">
                      Select Image
                    </button>
                  </div>
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
              <div className="p-5 bg-red-50 border-2 border-red-300 rounded-2xl shadow-md">
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                  <p className="text-sm font-semibold text-red-900">{error}</p>
                </div>
              </div>
            )}
          </div>

          {/* Right: Results */}
          <div className="space-y-6">
            {result && (
              <div className="animate-in fade-in zoom-in-95 duration-500 space-y-6">
                {/* Main Result Card */}
                <div className={`p-8 rounded-3xl border-2 ${activeMeta.bg} ${activeMeta.border} shadow-xl bg-white`}>
                  <div className="mb-8">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-blue-100 rounded-full mb-4 border border-blue-200">
                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: activeMeta.color }}></div>
                      <p className="text-xs font-bold text-blue-900 uppercase">Result</p>
                    </div>
                    <h2 className="text-5xl font-bold text-gray-900 mb-2" style={{ color: activeMeta.color }}>
                      {result.label}
                    </h2>
                    <div className="w-16 h-1 rounded-full" style={{ backgroundColor: activeMeta.color }}></div>
                  </div>

                  <p className="text-gray-700 mb-8 leading-relaxed text-base font-medium">
                    {result.label === "No DR" && "✓ No diabetic retinopathy detected. Continue routine monitoring."}
                    {result.label === "Mild DR" && "⚠ Early signs detected. Recommend follow-up and glycemic control."}
                    {result.label === "Moderate DR" && "⚠ Moderate changes present. Close monitoring recommended."}
                    {result.label === "Severe DR" && "🚨 Severe findings detected. Urgent evaluation needed."}
                    {result.label === "Proliferative DR" && "🚨 Advanced stage. Immediate intervention required."}
                  </p>

                  {/* Grade Badge */}
                  <div className="flex items-center gap-4 mb-8 pb-8 border-b-2 border-gray-200">
                    <div className={`px-4 py-2 rounded-full ${activeMeta.badge} ${activeMeta.badgeText} text-sm font-bold`}>
                      Grade {result.grade}
                    </div>
                    <span className="text-xs text-gray-600 font-bold uppercase">Severity</span>
                  </div>

                  {/* Confidence */}
                  <div>
                    <div className="flex justify-between items-baseline mb-3">
                      <p className="text-xs font-bold text-gray-700 uppercase">Confidence</p>
                      <p className="text-3xl font-bold" style={{ color: activeMeta.color }}>{result.confidence.toFixed(1)}%</p>
                    </div>
                    <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden shadow-inner">
                      <div
                        className="h-full rounded-full transition-all duration-1000 shadow-md"
                        style={{ width: `${result.confidence}%`, backgroundColor: activeMeta.color }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* Grade Reference */}
                <div className="bg-white p-6 rounded-2xl border-2 border-blue-200 shadow-lg">
                  <p className="text-xs font-bold text-blue-900 uppercase mb-5">
                    Severity Scale
                  </p>
                  <div className="grid grid-cols-5 gap-3">
                    {GRADE_META.map((grade, idx) => (
                      <div key={idx} className="text-center group cursor-pointer transition-all">
                        <div
                          className={`w-full p-3 rounded-lg text-white font-bold text-sm mb-2 transition-all shadow-md ${
                            result.grade === idx ? "ring-2 ring-offset-2 scale-110" : "opacity-70 group-hover:opacity-90"
                          }`}
                          style={{ backgroundColor: grade.color }}
                        >
                          {idx}
                        </div>
                        <p className="text-xs text-gray-700 font-semibold leading-tight">
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
                  className="w-full py-4 bg-white border-2 border-blue-300 text-blue-700 font-bold rounded-xl hover:bg-blue-50 transition-all shadow-md"
                >
                  New Analysis
                </button>
              </div>
            )}

            {!result && (
              <div className="bg-white p-12 rounded-3xl border-2 border-blue-200 text-center text-gray-500 min-h-96 flex flex-col items-center justify-center shadow-lg">
                <div className="w-20 h-20 mb-6 rounded-2xl bg-blue-50 flex items-center justify-center border-2 border-blue-200">
                  <svg className="w-10 h-10 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  </svg>
                </div>
                <p className="text-xl font-semibold text-gray-900 mb-2">Ready for Analysis</p>
                <p className="text-gray-600">Upload a fundus image to begin</p>
              </div>
            )}
          </div>
        </div>

        {/* Clinical Notes Footer */}
        <div className="mt-16 p-7 bg-blue-50 border-2 border-blue-300 rounded-2xl shadow-lg">
          <div className="flex gap-4">
            <div className="text-2xl flex-shrink-0">⚠️</div>
            <div>
              <p className="text-sm font-bold text-blue-900 uppercase tracking-widest mb-2">Clinical Use Notice</p>
              <p className="text-gray-800 leading-relaxed">
                This AI-assisted tool is intended to support clinical decision-making only.</p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
