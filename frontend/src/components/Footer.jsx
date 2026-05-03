import { NavLink } from "react-router-dom";

export default function Footer() {
  const linkStyle = {
    textDecoration: "none",
    color: "#64748b",
    fontSize: "12px",
    fontFamily: "'JetBrains Mono', monospace",
  };

  return (
    <footer
      style={{
        marginTop: 40,
        borderTop: "1px solid #e8edf3",
        background: "#ffffff",
      }}
    >
      <div
        style={{
          maxWidth: 1100,
          margin: "0 auto",
          padding: "20px 24px",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        {/* Top Row */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 10,
          }}
        >
          {/* Brand */}
          <div>
            <span style={{ fontWeight: 700, fontSize: 13, color: "#0f172a" }}>
              EYE AI
            </span>
            <p
              style={{
                fontSize: 12,
                color: "#94a3b8",
                marginTop: 4,
                maxWidth: 280,
                lineHeight: 1.6,
              }}
            >
              AI-powered retinal disease detection using OCT and fundus imaging.
            </p>
          </div>

          {/* Links */}
          <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
            <NavLink to="/" style={linkStyle}>
              Home
            </NavLink>
            <NavLink to="/dr" style={linkStyle}>
              DR
            </NavLink>
            <NavLink to="/oct" style={linkStyle}>
              OCT
            </NavLink>
            <NavLink to="/glaucoma" style={linkStyle}>
              Glaucoma
            </NavLink>
            <NavLink to="/vds" style={linkStyle}>
              VDS
            </NavLink>
          </div>
        </div>

        {/* Bottom Row */}
        <div
          style={{
            borderTop: "1px solid #f1f5f9",
            paddingTop: 10,
            display: "flex",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 8,
          }}
        >
          <span
            style={{
              fontSize: 11,
              color: "#94a3b8",
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            © {new Date().getFullYear()} EYE AI System
          </span>
        </div>
      </div>
    </footer>
  );
}
