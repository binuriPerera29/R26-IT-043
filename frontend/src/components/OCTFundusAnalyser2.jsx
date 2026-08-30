/**
 * OCTFundusAnalyser.jsx
 * EYE OCT Retinal Disease Analysis System — Pure Tailwind CSS, zero custom styles
 * Includes integrated RiskMeter component
 */

import { useState, useRef, useCallback } from "react";
import { predictOCT } from "../services/api_oct.js";

// ── Class metadata ──────────────────────────────────────────────────────────
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

// ── Risk Meter Metadata ─────────────────────────────────────────────────────
const RISK_META = {
  NORMAL: {
    trackColor: "#22c55e",
    riskPct: 5,
    pill: "Low Risk",
    pillBg: "bg-emerald-50",
    pillText: "text-emerald-700",
    pillBorder: "border-emerald-200",
    urgencyBg: "bg-emerald-50",
    urgencyText: "text-emerald-700",
    accentBorder: "border-l-emerald-500",
    dotBg: "bg-emerald-500",
    urgency: "Routine monitoring — annual eye exam recommended",
    title: "Normal Retina — No pathology detected",
    body: "Retinal layers appear intact with no fluid, drusen, or neovascular membranes. Continue regular eye exams annually. Patients over 50 or with family history of AMD should be screened more frequently.",
    findings: [
      "Intact IS/OS junction",
      "No sub-retinal fluid",
      "Normal foveal contour",
    ],
  },
  DRUSEN: {
    trackColor: "#eab308",
    riskPct: 38,
    pill: "Moderate Risk",
    pillBg: "bg-yellow-50",
    pillText: "text-yellow-800",
    pillBorder: "border-yellow-200",
    urgencyBg: "bg-yellow-50",
    urgencyText: "text-yellow-800",
    accentBorder: "border-l-yellow-400",
    dotBg: "bg-yellow-500",
    urgency: "Monitor every 6–12 months — watch for AMD progression",
    title: "Drusen — Early-stage macular degeneration",
    body: "Yellow lipid deposits form under the retinal pigment epithelium. While vision is often preserved initially, large drusen significantly raise the risk of progressing to wet AMD or geographic atrophy over years.",
    findings: [
      "Sub-RPE deposits",
      "RPE irregularity",
      "No active neovascularization",
    ],
  },
  DME: {
    trackColor: "#f97316",
    riskPct: 68,
    pill: "High Risk",
    pillBg: "bg-orange-50",
    pillText: "text-orange-800",
    pillBorder: "border-orange-200",
    urgencyBg: "bg-orange-50",
    urgencyText: "text-orange-800",
    accentBorder: "border-l-orange-500",
    dotBg: "bg-orange-500",
    urgency: "Refer within 1–4 weeks — vision loss risk without treatment",
    title: "DME — Diabetic macular swelling",
    body: "Breakdown of the blood-retinal barrier in diabetic patients causes fluid to accumulate in the macula. Central vision can deteriorate rapidly. Anti-VEGF injections or laser therapy are first-line treatments. Systemic glucose control is essential.",
    findings: [
      "Intra-retinal cystoid spaces",
      "Sub-retinal fluid possible",
      "Disrupted ellipsoid zone",
    ],
  },
  CNV: {
    trackColor: "#ef4444",
    riskPct: 95,
    pill: "Critical Risk",
    pillBg: "bg-red-50",
    pillText: "text-red-800",
    pillBorder: "border-red-200",
    urgencyBg: "bg-red-50",
    urgencyText: "text-red-800",
    accentBorder: "border-l-red-500",
    dotBg: "bg-red-500",
    urgency: "Urgent referral within 48–72 hours — rapid vision loss possible",
    title: "CNV — Active neovascular membrane",
    body: "Abnormal blood vessels grow from the choroid into the sub-retinal space, leaking fluid and blood. This is the most dangerous OCT finding — untreated CNV can cause severe irreversible central vision loss within weeks. Immediate anti-VEGF therapy is required.",
    findings: [
      "Sub-retinal hyper-reflective membrane",
      "Sub-retinal / intra-retinal fluid",
      "Disrupted Bruch's membrane",
    ],
  },
};

// ── Icons ───────────────────────────────────────────────────────────────────
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

function AlertTriangleIcon({ size = 22, className = "" }) {
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

// ── OODResultView ────────────────────────────────────────────────────────
// Shown INSTEAD OF the prediction banner + medical content whenever the
// image is flagged out-of-distribution. The model's classification and
// explanation text are deliberately withheld here — an OOD flag means the
// model doesn't recognize what it's looking at, so surfacing a confident
// diagnosis banner would be misleading. Only the original image and the
// raw OOD numbers are shown.
function OODResultView({ result, onReset }) {
  const ood = result.ood_detection;
  const gaugeMax = Math.max(ood.score * 1.4, ood.threshold * 1.4, 1);
  const scorePct = Math.min((ood.score / gaugeMax) * 100, 100);
  const threshPct = Math.min((ood.threshold / gaugeMax) * 100, 100);

  return (
    <div className="flex flex-col gap-3.5">
      {/* Banner — explicitly does not show a prediction label */}
      <div className="px-5 py-4 rounded-2xl border bg-red-50 border-red-200 flex items-start gap-3.5">
        <div className="flex items-center justify-center flex-shrink-0 bg-white border border-red-200 w-11 h-11 rounded-xl">
          <AlertTriangleIcon size={20} className="text-red-500" />
        </div>
        <div className="flex-1">
          <span className="font-mono text-[9px] text-red-500 tracking-widest uppercase block mb-1">
            Out-of-Distribution — Classification Withheld
          </span>
          <h2 className="mb-1 text-lg font-bold tracking-tight text-red-900">
            This image doesn't match the model's training distribution
          </h2>
          <p className="text-[13px] text-red-800 leading-relaxed">
            {result.warning ||
              "This image's feature embedding falls outside the model's training distribution. It may not be a genuine OCT scan, or may show pathology outside the four trained classes (CNV, DME, Drusen, Normal). No diagnosis is shown because a classification here would not be reliable."}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3.5 items-start max-lg:grid-cols-1">
        {/* LEFT: original image only — no Grad-CAM/overlay tabs, no risk meter */}
        <div className="bg-white border border-slate-200 rounded-2xl p-3.5">
          <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-2">
            Uploaded Image
          </span>
          <div className="flex items-center justify-center overflow-hidden border rounded-xl border-slate-200 bg-slate-50 aspect-square">
            <img
              src={`data:image/png;base64,${result.images.original}`}
              alt="Uploaded OCT scan"
              className="object-cover w-full h-full"
            />
          </div>
          <p className="text-[11px] text-slate-400 mt-2 text-center">
            Grad-CAM and class explanations are not generated for
            out-of-distribution images.
          </p>
        </div>

        {/* RIGHT: OOD detail panel */}
        <div className="px-5 py-4 bg-white border border-slate-200 rounded-2xl">
          <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-3.5">
            Out-of-Distribution Detection Details
          </span>

          <div className="flex items-center justify-between mb-3">
            <span className="text-[12px] text-slate-500">Detection method</span>
            <span className="text-[11px] font-mono text-slate-700">
              Mahalanobis distance
            </span>
          </div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[12px] text-slate-500">Status</span>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded border bg-red-50 text-red-700 border-red-200">
              {ood.status}
            </span>
          </div>

          {/* Gauge */}
          <div className="mt-4 mb-1">
            <div className="relative h-3 overflow-hidden rounded-full bg-slate-100">
              <div
                className="absolute inset-y-0 left-0 bg-red-500 rounded-full"
                style={{ width: `${scorePct}%` }}
              />
              <div
                className="absolute top-0 bottom-0 w-0.5 bg-slate-900"
                style={{ left: `${threshPct}%` }}
              />
            </div>
            <div className="flex justify-between font-mono text-[9px] text-slate-400 mt-1">
              <span>0</span>
              <span>{gaugeMax.toFixed(1)}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2.5 mt-3">
            <div className="rounded-xl bg-red-50 border border-red-200 px-3 py-2.5">
              <span className="font-mono text-[9px] text-red-500 tracking-widest uppercase block mb-1">
                OOD Score
              </span>
              <span className="font-mono text-lg font-bold text-red-700">
                {ood.score.toFixed(2)}
              </span>
            </div>
            <div className="rounded-xl bg-slate-50 border border-slate-200 px-3 py-2.5">
              <span className="font-mono text-[9px] text-slate-400 tracking-widest uppercase block mb-1">
                Threshold
              </span>
              <span className="font-mono text-lg font-bold text-slate-700">
                {ood.threshold.toFixed(2)}
              </span>
            </div>
          </div>

          <div className="mt-3.5 pt-3.5 border-t border-slate-100 text-[11px] text-slate-500 leading-relaxed">
            The score is the image's minimum Mahalanobis distance to any of
            the four trained class centroids in embedding space. It exceeds
            the threshold calibrated on held-out validation data, so this
            image is treated as not represented by the training set.
          </div>

          <button
            onClick={onReset}
            className="mt-4 w-full h-9 rounded-lg border border-slate-200 bg-white text-slate-600 text-[13px] font-medium cursor-pointer hover:border-slate-300 hover:text-slate-800 transition-all"
          >
            Try a different image
          </button>
        </div>
      </div>
    </div>
  );
}

// ── CautionBanner ─────────────────────────────────────────────────────────
// Low-confidence-only caution (OOD has its own dedicated view above, so this
// only fires for the "in-distribution but not confident" case).
function CautionBanner({ result }) {
  const conf = result.confidence_check;
  if (!conf?.is_low_confidence) return null;

  return (
    <div className="flex items-start gap-3 px-5 py-4 border bg-amber-50 border-amber-200 rounded-2xl">
      <span className="flex-shrink-0 text-xl">⚠️</span>
      <div className="flex-1">
        <span className="font-mono text-[10px] tracking-widest uppercase text-amber-600 block mb-1">
          Caution — Low Confidence Prediction
        </span>
        <p className="text-[13px] text-amber-800 leading-relaxed mb-2">
          {result.warning}
        </p>
        <span className="text-[10px] font-mono px-2.5 py-1 rounded-lg border bg-amber-100 text-amber-800 border-amber-300">
          Confidence: {conf.confidence.toFixed(1)}% · min required {conf.threshold.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

// ── DiagnosticsPanel ────────────────────────────────────────────────────────
function DiagnosticsRow({ label, status, isBad, detail }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-100 last:border-b-0">
      <span className="text-[11px] text-slate-500">{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono text-slate-400">{detail}</span>
        <span
          className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${
            isBad
              ? "bg-red-50 text-red-600 border-red-200"
              : "bg-emerald-50 text-emerald-600 border-emerald-200"
          }`}
        >
          {status}
        </span>
      </div>
    </div>
  );
}

function DiagnosticsPanel({ result }) {
  const ood = result.ood_detection;
  const conf = result.confidence_check;
  if (!ood && !conf) return null;

  return (
    <div className="bg-white border border-slate-200 rounded-xl px-4 py-3.5">
      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-1">
        Detection Diagnostics
      </span>
      <div className="flex flex-col">
        {ood && (
          <DiagnosticsRow
            label="Out-of-Distribution Check"
            status={ood.status}
            isBad={ood.is_ood}
            detail={`${ood.score.toFixed(2)} / ${ood.threshold.toFixed(2)}`}
          />
        )}
        {conf && (
          <DiagnosticsRow
            label="Confidence Check"
            status={conf.status}
            isBad={conf.is_low_confidence}
            detail={`${conf.confidence.toFixed(1)}% / min ${conf.threshold.toFixed(0)}%`}
          />
        )}
      </div>
    </div>
  );
}

// ── RiskMeter ───────────────────────────────────────────────────────────────
function RiskMeter({ activeCondition = null }) {
  const [selected, setSelected] = useState(activeCondition || "CNV");

  // Sync with parent when activeCondition changes (after scan result arrives)
  const resolvedSelected =
    activeCondition !== null ? activeCondition : selected;
  const handleSelect = activeCondition !== null ? undefined : setSelected;

  const c = RISK_META[resolvedSelected];
  const classMeta = CLASS_META[resolvedSelected];

  return (
    <div className="p-5 bg-white border border-slate-200 rounded-2xl">
      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-3">
        Clinical Risk Level
      </span>

      {/* Condition selector cards */}
      <div className="grid grid-cols-4 gap-2 mb-4">
        {CLASS_ORDER.map((key) => {
          const m = CLASS_META[key];
          const r = RISK_META[key];
          const isActive = resolvedSelected === key;
          return (
            <button
              key={key}
              onClick={() => handleSelect && handleSelect(key)}
              className={`rounded-xl p-3 text-left border cursor-pointer transition-all
                ${m.bg} ${m.border}
                ${isActive ? "ring-2 ring-offset-1" : "opacity-50 hover:opacity-80"}
                ${activeCondition !== null ? "cursor-default" : ""}`}
              style={isActive ? { "--tw-ring-color": r.trackColor } : {}}
            >
              <div
                className={`font-mono text-[12px] font-bold mb-0.5 ${m.color}`}
              >
                {key}
              </div>
              <div className="text-[10px] text-slate-500 leading-snug mb-2">
                {m.label}
              </div>
              <span
                className={`text-[9px] font-mono px-1.5 py-0.5 rounded border ${r.pillBg} ${r.pillText} ${r.pillBorder}`}
              >
                {r.pill}
              </span>
            </button>
          );
        })}
      </div>

      {/* Risk gradient track */}
      <div className="relative mb-1">
        <div
          className="h-3 rounded-full"
          style={{
            background:
              "linear-gradient(to right, #22c55e 0%, #eab308 40%, #f97316 70%, #ef4444 100%)",
          }}
        />
        {/* Needle */}
        <div
          className="absolute w-5 h-5 transition-all duration-700 -translate-x-1/2 -translate-y-1/2 bg-white border-2 rounded-full shadow-sm top-1/2"
          style={{ left: `${c.riskPct}%`, borderColor: c.trackColor }}
        />
      </div>
      <div className="flex justify-between font-mono text-[9px] text-slate-400 mb-3">
        {["Low", "Moderate", "High", "Critical"].map((l) => (
          <span key={l}>{l}</span>
        ))}
      </div>

      {/* Urgency bar */}
      <div
        className={`rounded-xl px-3 py-2 font-mono text-[11px] font-medium mb-3 ${c.urgencyBg} ${c.urgencyText}`}
      >
        {c.urgency}
      </div>

      {/* Detail panel */}
      <div
        className={`rounded-xl border-l-4 bg-slate-50 px-4 py-3 ${c.accentBorder}`}
      >
        <p className="font-mono text-[10px] tracking-widest uppercase text-slate-400 mb-1">
          {c.title}
        </p>
        <p className="text-[12px] text-slate-500 leading-relaxed mb-2">
          {c.body}
        </p>
        <div className="flex flex-col gap-1">
          {c.findings.map((f, i) => (
            <div key={i} className="flex items-center gap-2 text-[11px] text-slate-500">
              <span
                className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${c.dotBg}`}
              />
              {f}
            </div>
          ))}
        </div>
      </div>
    </div>
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

  const isOOD = result?.ood_detection?.is_ood === true;

  return (
    <div className="min-h-screen w-full bg-[#f6f7f9] [background-image:radial-gradient(#d1d5db_1px,transparent_1px)] [background-size:22px_22px]">
      {/* ── Navbar ── */}
      <nav className="w-full bg-white border-b border-slate-200">
        <div className="max-w-[1400px] mx-auto px-8 flex items-center justify-between h-14">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center border border-indigo-200 rounded-lg w-7 h-7 bg-gradient-to-br from-indigo-50 to-indigo-100">
              <EyeIcon size={14} className="text-indigo-500" />
            </div>
            <div>
              <span className="font-bold text-[13px] text-slate-900 tracking-tight">
                EYE OCT
              </span>
              <span className="font-mono text-[9px] text-slate-400 block tracking-widest">
                RETINAL ANALYSIS SYSTEM
              </span>
            </div>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span className="font-mono text-[9px] text-emerald-500 tracking-widest">
              ONLINE
            </span>
          </div>
        </div>
      </nav>

      {/* ── Content ── */}
      <div className="max-w-[1400px] mx-auto px-8 py-8 pb-16">
        {/* ── Hero ── */}
        <div className="text-center mb-6 animate-[fadeUp_0.5s_ease_both]">
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
                    className="inline-flex items-center gap-1.5 h-9 px-4 rounded-lg border-0 bg-gradient-to-br from-yellow-400 to-amber-500 text-slate-900 text-[13px] font-semibold cursor-pointer shadow-lg shadow-yellow-200 hover:shadow-xl hover:shadow-yellow-300 hover:-translate-y-px transition-all disabled:bg-slate-100 disabled:text-slate-300 disabled:shadow-none disabled:cursor-not-allowed"
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

        {/* ── Empty state: class reference cards + risk meter ── */}
        {!result && !loading && (
          <div>
            <p className="text-center font-mono text-[10px] text-slate-400 tracking-widest uppercase mb-3">
              Detectable OCT Conditions
            </p>
            <div className="grid max-w-4xl grid-cols-4 gap-4 mx-auto mb-8">
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

            {/* Risk meter in empty state — interactive, no locked condition */}
            <div className="max-w-4xl mx-auto">
              <RiskMeter activeCondition={null} />
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {result && (
          <div
            ref={resultRef}
            className="animate-[fadeUp_0.5s_cubic-bezier(0.22,1,0.36,1)_both]"
          >
            {isOOD ? (
              // ── OOD path: original image + OOD details only.
              // No prediction label, no medical content, no risk meter.
              <OODResultView result={result} onReset={handleReset} />
            ) : (
              // ── Normal path: full prediction + explanation content.
              <div className="flex flex-col gap-3.5">
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
                          <span
                            className={`text-xl font-mono font-bold ${meta.color}`}
                          >
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
                        <span
                          className={`font-mono text-[46px] font-bold leading-none ${meta.color}`}
                        >
                          {result.confidence.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })()}

                {/* Caution banner — low-confidence only (OOD has its own view) */}
                <CautionBanner result={result} />

                {/* Two-col layout */}
                <div className="grid grid-cols-[1.2fr_0.9fr] gap-3.5 items-start max-lg:grid-cols-1">
                  {/* LEFT: image-focused analysis */}
                  <div className="flex flex-col gap-2.5">
                    <ImageViewer
                      result={result}
                      activeTab={activeTab}
                      setActiveTab={setActiveTab}
                    />

                    <div className="bg-white border border-slate-200 border-l-4 border-l-indigo-500 rounded-xl px-4 py-3.5">
                      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-2">
                        Grad-CAM Interpretation
                      </span>
                      <p className="text-[12px] text-slate-500 leading-relaxed">
                        {result.medical_explanation.gradcam_interpretation}
                      </p>
                      <div className="flex gap-2.5 mt-2 flex-wrap">
                        {[
                          ["bg-red-500", "High activation"],
                          ["bg-amber-500", "Moderate"],
                          ["bg-blue-500", "Low activation"],
                        ].map(([c, l]) => (
                          <span
                            key={l}
                            className="flex items-center gap-1 text-[10px] font-mono text-slate-400"
                          >
                            <span
                              className={`w-2.5 h-1 rounded-sm inline-block ${c}`}
                            />
                            {l}
                          </span>
                        ))}
                      </div>
                    </div>

                    <DiagnosticsPanel result={result} />

                    <RiskMeter activeCondition={result.prediction} />
                  </div>

                  {/* RIGHT: summary and medical context */}
                  <div className="flex flex-col gap-2.5">
                    <div className="px-5 py-4 bg-white border border-slate-200 rounded-2xl">
                      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-3.5">
                        Class Probabilities
                      </span>
                      <div className="flex flex-col gap-3.5">
                        {CLASS_ORDER.slice()
                          .sort(
                            (a, b) =>
                              result.class_probabilities[b] -
                              result.class_probabilities[a],
                          )
                          .map((cls) => (
                            <ProbabilityBar
                              key={cls}
                              cls={cls}
                              value={result.class_probabilities[cls]}
                              isTop={cls === topClass}
                            />
                          ))}
                      </div>
                    </div>

                    <div className="flex items-center gap-2.5">
                      <div className="flex-1 h-px bg-slate-200" />
                      <span className="font-mono text-[10px] text-indigo-500 flex items-center gap-1 whitespace-nowrap">
                        🩺 Medical Explanation
                      </span>
                      <div className="flex-1 h-px bg-slate-200" />
                    </div>

                    <div className="bg-white border border-slate-200 rounded-xl px-4 py-3.5">
                      <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-2">
                        About This Condition
                      </span>
                      <p className="text-[12px] text-slate-500 leading-relaxed">
                        {result.medical_explanation.description}
                      </p>
                    </div>

                    {(result.medical_explanation.urgency ||
                      result.medical_explanation.prognosis) &&
                      (() => {
                        const meta =
                          CLASS_META[result.prediction] || CLASS_META.NORMAL;
                        return (
                          <div className="grid grid-cols-2 gap-2.5">
                            {result.medical_explanation.urgency && (
                              <div
                                className={`bg-white border border-slate-200 border-l-4 rounded-xl px-4 py-3.5 ${meta.accent}`}
                              >
                                <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-1.5">
                                  Clinical Urgency
                                </span>
                                <p className="text-[12px] text-slate-700 leading-relaxed">
                                  {result.medical_explanation.urgency}
                                </p>
                              </div>
                            )}
                            {result.medical_explanation.prognosis && (
                              <div className="bg-white border border-slate-200 border-l-4 border-l-emerald-500 rounded-xl px-4 py-3.5">
                                <span className="font-mono text-[10px] tracking-widest uppercase text-slate-400 block mb-1.5">
                                  Prognosis
                                </span>
                                <p className="text-[12px] text-slate-700 leading-relaxed">
                                  {result.medical_explanation.prognosis}
                                </p>
                              </div>
                            )}
                          </div>
                        );
                      })()}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}