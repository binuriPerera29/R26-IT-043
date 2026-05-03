import { NavLink, useLocation } from "react-router-dom";
import { useState, useEffect } from "react";

const navItems = [
  { to: "/", label: "Home", exact: true },
  {
    to: "/dr",
    label: "Diabetic Retinopathy",
    short: "DR",
    color: "#1a6fc4",
    dot: "#60a5fa",
  },
  {
    to: "/oct",
    label: "OCT Analysis",
    short: "OCT",
    color: "#0e8a6e",
    dot: "#34d399",
  },
  {
    to: "/glaucoma",
    label: "Glaucoma",
    short: "GLC",
    color: "#8b3fbd",
    dot: "#a78bfa",
  },
  {
    to: "/cataract",
    label: "Cataract",
    short: "CAT",
    color: "#c4720e",
    dot: "#fbbf24",
  },
];

function Logo() {
  return (
    <NavLink
      to="/"
      style={{ textDecoration: "none", display: "flex", alignItems: "center", gap: 10 }}
    >
      {/* Icon mark */}
      <div
        style={{
          width: 34,
          height: 34,
          borderRadius: 9,
          background: "linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          boxShadow: "0 2px 8px rgba(26,111,196,0.25)",
        }}
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
          {/* Outer eye shape */}
          <path
            d="M2 10 C5 5.5 15 5.5 18 10 C15 14.5 5 14.5 2 10Z"
            stroke="white"
            strokeWidth="1.3"
            fill="none"
            strokeLinejoin="round"
          />
          {/* Iris */}
          <circle cx="10" cy="10" r="3.2" stroke="white" strokeWidth="1.2" fill="none" />
          {/* Pupil */}
          <circle cx="10" cy="10" r="1.3" fill="white" opacity="0.9" />
          {/* Scan lines */}
          <line x1="2" y1="10" x2="5.5" y2="10" stroke="#60a5fa" strokeWidth="0.9" opacity="0.7" />
          <line x1="14.5" y1="10" x2="18" y2="10" stroke="#60a5fa" strokeWidth="0.9" opacity="0.7" />
          {/* Light reflection */}
          <circle cx="11.4" cy="8.8" r="0.7" fill="white" opacity="0.6" />
        </svg>
      </div>

      {/* Wordmark */}
      <div style={{ display: "flex", flexDirection: "column", lineHeight: 1 }}>
        <span
          style={{
            fontSize: 15,
            fontWeight: 750,
            color: "#0f172a",
            letterSpacing: "-0.5px",
            fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
          }}
        >
          Retinal<span style={{ color: "#1a6fc4" }}>AI</span>
        </span>
        <span
          style={{
            fontSize: 9.5,
            color: "#94a3b8",
            letterSpacing: "1.2px",
            fontWeight: 550,
            textTransform: "uppercase",
            marginTop: 1,
          }}
        >
          Eye Disease Detection
        </span>
      </div>
    </NavLink>
  );
}

export default function NavBar() {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 8);
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  // Close menu on route change
  useEffect(() => setMenuOpen(false), [location.pathname]);

  const isActive = (to, exact) =>
    exact ? location.pathname === to : location.pathname.startsWith(to);

  const activeItem = navItems.find(
    (n) => n.to !== "/" && location.pathname.startsWith(n.to)
  );

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

        .nav-link-pill {
          position: relative;
          display: flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          font-weight: 550;
          text-decoration: none;
          padding: 6px 13px;
          border-radius: 8px;
          color: #475569;
          background: transparent;
          transition: color 0.18s, background 0.18s;
          white-space: nowrap;
          font-family: 'DM Sans', 'Helvetica Neue', sans-serif;
        }
        .nav-link-pill:hover {
          color: #0f172a;
          background: #f1f5f9;
        }
        .nav-link-pill.active {
          color: var(--nav-accent, #1a6fc4);
          background: var(--nav-accent-bg, #e8f1fc);
          font-weight: 650;
        }
        .nav-dot {
          width: 6px;
          height: 6px;
          border-radius: 50%;
          flex-shrink: 0;
        }
        .mobile-menu-btn {
          display: none;
          background: none;
          border: none;
          cursor: pointer;
          padding: 6px;
          border-radius: 8px;
          color: #475569;
        }
        @media (max-width: 760px) {
          .nav-links-desktop { display: none !important; }
          .mobile-menu-btn { display: flex !important; }
          .nav-right-desktop { display: none !important; }
        }
      `}</style>

      <nav
        style={{
          background: "#ffffff",
          borderBottom: scrolled ? "1px solid #d8dce3" : "1px solid #e8eaed",
          width: "100%",
          position: "sticky",
          top: 0,
          zIndex: 1000,
          boxShadow: scrolled ? "0 2px 12px rgba(0,0,0,0.06)" : "none",
          transition: "box-shadow 0.2s, border-color 0.2s",
          fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif",
        }}
      >
        {/* Active module indicator bar */}
        {activeItem && (
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              height: 2.5,
              background: `linear-gradient(90deg, ${activeItem.color} 0%, ${activeItem.dot} 100%)`,
              transition: "opacity 0.3s",
            }}
          />
        )}

        <div
          style={{
            maxWidth: 1100,
            margin: "0 auto",
            padding: "0 24px",
            height: 56,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 16,
          }}
        >
          {/* Logo */}
          <Logo />

          {/* Desktop nav links */}
          <div
            className="nav-links-desktop"
            style={{ display: "flex", alignItems: "center", gap: 2, flex: 1, justifyContent: "center" }}
          >
            {/* Divider */}
            <div style={{ width: 1, height: 20, background: "#e2e5ea", marginRight: 10 }} />

            {navItems.map((item) => {
              const active = isActive(item.to, item.exact);
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.exact}
                  className="nav-link-pill"
                  style={{
                    "--nav-accent": item.color || "#0f172a",
                    "--nav-accent-bg": item.color ? item.color + "14" : "#f1f5f9",
                  }}
                >
                  {item.dot && active && (
                    <span
                      className="nav-dot"
                      style={{ background: item.dot }}
                    />
                  )}
                  {item.to === "/" ? item.label : (
                    <>
                      <span className="link-short" style={{ display: "none" }}>{item.short}</span>
                      <span className="link-full">{item.label}</span>
                    </>
                  )}
                </NavLink>
              );
            })}
          </div>

          {/* Right side */}
          <div
            className="nav-right-desktop"
            style={{ display: "flex", alignItems: "center", gap: 10 }}
          >



          </div>

          {/* Mobile hamburger */}
          <button
            className="mobile-menu-btn"
            onClick={() => setMenuOpen((o) => !o)}
            aria-label="Toggle menu"
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              {menuOpen ? (
                <>
                  <line x1="4" y1="4" x2="16" y2="16" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                  <line x1="16" y1="4" x2="4" y2="16" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </>
              ) : (
                <>
                  <line x1="3" y1="6" x2="17" y2="6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                  <line x1="3" y1="10" x2="17" y2="10" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                  <line x1="3" y1="14" x2="17" y2="14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
                </>
              )}
            </svg>
          </button>
        </div>

        {/* Mobile dropdown */}
        {menuOpen && (
          <div
            style={{
              borderTop: "1px solid #e8eaed",
              background: "#fff",
              padding: "12px 20px 16px",
              display: "flex",
              flexDirection: "column",
              gap: 4,
            }}
          >
            {navItems.map((item) => {
              const active = isActive(item.to, item.exact);
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.exact}
                  className="nav-link-pill"
                  style={{
                    "--nav-accent": item.color || "#0f172a",
                    "--nav-accent-bg": item.color ? item.color + "14" : "#f1f5f9",
                    justifyContent: "flex-start",
                    padding: "9px 14px",
                  }}
                >
                  {item.dot && (
                    <span
                      className="nav-dot"
                      style={{ background: active ? item.dot : "#cbd5e1" }}
                    />
                  )}
                  {item.label}
                </NavLink>
              );
            })}

            <div style={{ marginTop: 8, paddingTop: 10, borderTop: "1px solid #f0f2f5" }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  padding: "6px 14px",
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#22c55e",
                    display: "inline-block",
                  }}
                />
                <span style={{ fontSize: 12, color: "#15803d", fontWeight: 550 }}>
                  All models online
                </span>
              </div>
            </div>
          </div>
        )}
      </nav>
    </>
  );
}