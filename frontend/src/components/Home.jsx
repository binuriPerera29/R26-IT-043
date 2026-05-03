import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const modules = [
  {
    id: "dr",
    path: "/dr",
    title: "Diabetic Retinopathy",
    abbr: "DR",
    tagline: "Fundus · Deep Learning",
    desc: "Detect and grade DR stages (No DR → Proliferative) from retinal fundus photographs using CNN-based classification.",
    stats: { label: "Stages Detected", value: "5" },
    accent: "#1a6fc4",
    light: "#e8f1fc",
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <circle
          cx="14"
          cy="14"
          r="11"
          stroke="currentColor"
          strokeWidth="1.6"
        />
        <circle cx="14" cy="14" r="4.5" fill="currentColor" opacity="0.15" />
        <circle cx="14" cy="14" r="2" fill="currentColor" />
        <line
          x1="14"
          y1="3"
          x2="14"
          y2="6.5"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="14"
          y1="21.5"
          x2="14"
          y2="25"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="3"
          y1="14"
          x2="6.5"
          y2="14"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
        <line
          x1="21.5"
          y1="14"
          x2="25"
          y2="14"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
    ),
  },
  {
    id: "oct",
    path: "/oct",
    title: "OCT Analysis",
    abbr: "OCT",
    tagline: "Retinal Layers · Cross-section",
    desc: "Classify OCT scans into CNV, DME, Drusen and Normal categories using advanced image segmentation models.",
    stats: { label: "Classifications", value: "4" },
    accent: "#0e8a6e",
    light: "#e3f5f0",
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <rect
          x="3"
          y="7"
          width="22"
          height="14"
          rx="2"
          stroke="currentColor"
          strokeWidth="1.6"
        />
        <path
          d="M3 12 Q8 9 14 12 Q20 15 25 12"
          stroke="currentColor"
          strokeWidth="1.4"
          fill="none"
          strokeLinecap="round"
        />
        <path
          d="M3 16 Q8 13 14 16 Q20 19 25 16"
          stroke="currentColor"
          strokeWidth="1.4"
          fill="none"
          strokeLinecap="round"
        />
        <circle cx="14" cy="14" r="2" fill="currentColor" opacity="0.3" />
      </svg>
    ),
  },
  {
    id: "glaucoma",
    path: "/glaucoma",
    title: "Glaucoma Detection",
    abbr: "GLC",
    tagline: "Optic Nerve · Cup-to-Disc",
    desc: "Assess optic nerve head damage and cup-to-disc ratio to identify glaucoma risk levels from fundus images.",
    stats: { label: "Risk Levels", value: "3" },
    accent: "#8b3fbd",
    light: "#f0e8fa",
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <circle
          cx="14"
          cy="14"
          r="11"
          stroke="currentColor"
          strokeWidth="1.6"
        />
        <ellipse
          cx="14"
          cy="14"
          rx="5"
          ry="6.5"
          stroke="currentColor"
          strokeWidth="1.4"
        />
        <ellipse
          cx="14"
          cy="14"
          rx="2.5"
          ry="3.5"
          fill="currentColor"
          opacity="0.2"
          stroke="currentColor"
          strokeWidth="1.2"
        />
        <line
          x1="9"
          y1="14"
          x2="6"
          y2="14"
          stroke="currentColor"
          strokeWidth="1.3"
          strokeLinecap="round"
        />
        <line
          x1="19"
          y1="14"
          x2="22"
          y2="14"
          stroke="currentColor"
          strokeWidth="1.3"
          strokeLinecap="round"
        />
      </svg>
    ),
  },
  {
    id: "cataract",
    path: "/cataract",
    title: "Cataract Detection",
    abbr: "CAT",
    tagline: "Lens Opacity · Grading",
    desc: "Identify lens opacities and grade cataract severity to aid in surgical planning and clinical decision support.",
    stats: { label: "Severity Grades", value: "4" },
    accent: "#c4720e",
    light: "#fdf0e0",
    icon: (
      <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
        <ellipse
          cx="14"
          cy="14"
          rx="11"
          ry="8"
          stroke="currentColor"
          strokeWidth="1.6"
        />
        <circle cx="14" cy="14" r="5" stroke="currentColor" strokeWidth="1.4" />
        <circle cx="14" cy="14" r="2.5" fill="currentColor" opacity="0.25" />
        <circle cx="11.5" cy="11.5" r="1" fill="currentColor" opacity="0.4" />
        <circle cx="16.5" cy="12.5" r="0.7" fill="currentColor" opacity="0.3" />
        <circle cx="13" cy="16" r="0.8" fill="currentColor" opacity="0.35" />
      </svg>
    ),
  },
];

const steps = [
  { num: "01", label: "Upload Image", detail: "Fundus photo or OCT scan" },
  { num: "02", label: "AI Analysis", detail: "Deep learning inference" },
  { num: "03", label: "Review Results", detail: "Graded report with heatmap" },
];

export default function Home() {
  const navigate = useNavigate();
  const [hovered, setHovered] = useState(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVisible(true), 60);
    return () => clearTimeout(t);
  }, []);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f7f8fa",
        fontFamily: "'DM Sans', 'Helvetica Neue', Arial, sans-serif",
        opacity: visible ? 1 : 0,
        transition: "opacity 0.5s ease",
      }}
    >


      <main
        style={{ maxWidth: 1080, margin: "0 auto", padding: "48px 24px 80px" }}
      >
        {/* ── Hero ── */}
        <div
          style={{
            background: "#fff",
            borderRadius: 20,
            border: "1px solid #e8eaed",
            padding: "52px 56px",
            marginBottom: 32,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            overflow: "hidden",
            position: "relative",
          }}
        >
          {/* Decorative blob */}
          <div
            style={{
              position: "absolute",
              right: -60,
              top: -60,
              width: 320,
              height: 320,
              borderRadius: "50%",
              background:
                "radial-gradient(circle, #e8f1fc 0%, transparent 70%)",
              pointerEvents: "none",
            }}
          />
          <div style={{ position: "relative", maxWidth: 540 }}>
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                background: "#e8f1fc",
                color: "#1a6fc4",
                fontSize: 12,
                fontWeight: 600,
                padding: "4px 12px",
                borderRadius: 20,
                marginBottom: 18,
                letterSpacing: "0.3px",
              }}
            >
              <span
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: "50%",
                  background: "#1a6fc4",
                  display: "inline-block",
                }}
              />
              AI-Powered · Clinical Grade
            </div>
            <h1
              style={{
                fontSize: 36,
                fontWeight: 700,
                color: "#0f172a",
                lineHeight: 1.2,
                margin: "0 0 16px",
                letterSpacing: "-0.8px",
              }}
            >
              Eye Disease
              <br />
              <span style={{ color: "#1a6fc4" }}>Detection System</span>
            </h1>
            <p
              style={{
                fontSize: 15,
                color: "#64748b",
                lineHeight: 1.7,
                margin: "0 0 28px",
                maxWidth: 420,
              }}
            >
              Upload retinal fundus or OCT images for instant AI-assisted
              screening across four major ophthalmic conditions.
            </p>
            <div style={{ display: "flex", gap: 12 }}>
              <button
                onClick={() => navigate("/dr")}
                style={{
                  background: "#0f172a",
                  color: "#fff",
                  border: "none",
                  borderRadius: 10,
                  padding: "11px 24px",
                  fontSize: 14,
                  fontWeight: 600,
                  cursor: "pointer",
                  letterSpacing: "-0.1px",
                }}
              >
                Start Analysis →
              </button>
              
            </div>
          </div>

          {/* Stats cluster */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 14,
              position: "relative",
            }}
          >
            {[
              { val: "4", sub: "Disease modules" },
              { val: "98%", sub: "Model accuracy" },
              { val: "<2s", sub: "Inference time" },
              { val: "DICOM", sub: "Format support" },
            ].map((s, i) => (
              <div
                key={i}
                style={{
                  background: "#f7f8fa",
                  borderRadius: 14,
                  padding: "18px 20px",
                  minWidth: 120,
                  border: "1px solid #eef0f3",
                }}
              >
                <p
                  style={{
                    fontSize: 26,
                    fontWeight: 700,
                    color: "#0f172a",
                    margin: "0 0 4px",
                    letterSpacing: "-1px",
                  }}
                >
                  {s.val}
                </p>
                <p
                  style={{
                    fontSize: 12,
                    color: "#94a3b8",
                    margin: 0,
                    fontWeight: 450,
                  }}
                >
                  {s.sub}
                </p>
              </div>
            ))}
          </div>
        </div>

        {/* ── Module Cards ── */}
        <p
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: "#94a3b8",
            letterSpacing: "0.8px",
            textTransform: "uppercase",
            marginBottom: 14,
          }}
        >
          Detection Modules
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
            gap: 16,
            marginBottom: 36,
          }}
        >
          {modules.map((mod) => {
            const isHovered = hovered === mod.id;
            return (
              <div
                key={mod.id}
                onClick={() => navigate(mod.path)}
                onMouseEnter={() => setHovered(mod.id)}
                onMouseLeave={() => setHovered(null)}
                style={{
                  background: isHovered ? mod.light : "#fff",
                  border: `1px solid ${isHovered ? mod.accent + "40" : "#e8eaed"}`,
                  borderRadius: 16,
                  padding: "24px 22px",
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                  transform: isHovered ? "translateY(-4px)" : "none",
                  boxShadow: isHovered ? `0 12px 32px ${mod.accent}18` : "none",
                  display: "flex",
                  flexDirection: "column",
                  gap: 0,
                }}
              >
                {/* Icon + abbr */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "flex-start",
                    justifyContent: "space-between",
                    marginBottom: 16,
                  }}
                >
                  <div
                    style={{
                      width: 48,
                      height: 48,
                      borderRadius: 12,
                      background: isHovered ? mod.accent + "18" : "#f4f5f7",
                      color: isHovered ? mod.accent : "#64748b",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      transition: "all 0.2s",
                    }}
                  >
                    {mod.icon}
                  </div>
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      color: isHovered ? mod.accent : "#c0c7d0",
                      letterSpacing: "1px",
                      marginTop: 4,
                    }}
                  >
                    {mod.abbr}
                  </span>
                </div>

                <h3
                  style={{
                    fontSize: 15,
                    fontWeight: 650,
                    color: "#0f172a",
                    margin: "0 0 4px",
                    letterSpacing: "-0.2px",
                  }}
                >
                  {mod.title}
                </h3>
                <p
                  style={{
                    fontSize: 11,
                    color: mod.accent,
                    fontWeight: 550,
                    margin: "0 0 12px",
                    letterSpacing: "0.2px",
                  }}
                >
                  {mod.tagline}
                </p>
                <p
                  style={{
                    fontSize: 13,
                    color: "#64748b",
                    lineHeight: 1.6,
                    margin: "0 0 20px",
                    flexGrow: 1,
                  }}
                >
                  {mod.desc}
                </p>

                {/* Footer */}
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    borderTop: "1px solid #f0f2f5",
                    paddingTop: 14,
                  }}
                >
                  <div>
                    <p
                      style={{
                        fontSize: 18,
                        fontWeight: 700,
                        color: mod.accent,
                        margin: 0,
                        letterSpacing: "-0.5px",
                      }}
                    >
                      {mod.stats.value}
                    </p>
                    <p style={{ fontSize: 11, color: "#94a3b8", margin: 0 }}>
                      {mod.stats.label}
                    </p>
                  </div>
                  <div
                    style={{
                      width: 32,
                      height: 32,
                      borderRadius: "50%",
                      background: isHovered ? mod.accent : "#f4f5f7",
                      color: isHovered ? "#fff" : "#94a3b8",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 16,
                      transition: "all 0.2s",
                      transform: isHovered ? "translateX(2px)" : "none",
                    }}
                  >
                    →
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* ── How it works ── */}
        <div
          style={{
            background: "#fff",
            borderRadius: 16,
            border: "1px solid #e8eaed",
            padding: "32px 36px",
            marginBottom: 32,
          }}
        >
          <p
            style={{
              fontSize: 12,
              fontWeight: 600,
              color: "#94a3b8",
              letterSpacing: "0.8px",
              textTransform: "uppercase",
              margin: "0 0 8px",
            }}
          >
            Workflow
          </p>
          <h2
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: "#0f172a",
              margin: "0 0 28px",
              letterSpacing: "-0.4px",
            }}
          >
            How it works
          </h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: 0,
              position: "relative",
            }}
          >
            {steps.map((step, i) => (
              <div
                key={i}
                style={{ display: "flex", alignItems: "flex-start", gap: 0 }}
              >
                <div style={{ flex: 1 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      marginBottom: 10,
                    }}
                  >
                    <span
                      style={{
                        fontSize: 11,
                        fontWeight: 700,
                        color: "#1a6fc4",
                        background: "#e8f1fc",
                        borderRadius: 6,
                        padding: "3px 8px",
                        letterSpacing: "0.5px",
                      }}
                    >
                      {step.num}
                    </span>
                    {i < steps.length - 1 && (
                      <div
                        style={{
                          height: 1,
                          flex: 1,
                          background: "#e8eaed",
                          marginLeft: 4,
                        }}
                      />
                    )}
                  </div>
                  <p
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: "#0f172a",
                      margin: "0 0 4px",
                    }}
                  >
                    {step.label}
                  </p>
                  <p style={{ fontSize: 12, color: "#94a3b8", margin: 0 }}>
                    {step.detail}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>        
      </main>
    </div>
  );
}
