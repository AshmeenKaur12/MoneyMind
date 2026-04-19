import React from 'react';

const Navbar = () => {
  const openDiagnosticTool = () => {
    window.open("http://localhost:8501", "_blank");
  };
  return (
    <nav style={navStyles.navbar}>
      <div style={navStyles.container}>
        {/* Logo */}

        {/* Links */}
        <div style={navStyles.navLinks}>
          <a href="#home" style={navStyles.link}>Home</a>
          <a href="#about" style={navStyles.link}>About</a>
          <a href="#how-it-works" style={navStyles.link}>How it Works</a>
          <a href="#faq" style={navStyles.link}>Frequently Asked Questions</a>
          <button
            onClick={openDiagnosticTool}
            className="get-started-btn"
            style={navStyles.button}
          >
            Get Started
          </button>
        </div>
      </div>
    </nav>
  );
};

const navStyles = {
  navbar: {
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    backdropFilter: "blur(10px)",
    position: "sticky",
    top: 0,
    zIndex: 1000,
    borderBottom: "1px solid #e5e7eb",
    padding: "0.7rem 2rem",
  },
  container: {
    maxWidth: "1100px",
    margin: "0 auto",

    marginLeft: '32rem',

    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  logo: {
    fontSize: "1.5rem",
    fontWeight: "800",
    color: "#2d4a31",
    fontFamily: "'Inter', sans-serif",
  },
  navLinks: {
    display: "flex",

    gap: "2rem",
    alignItems: "center",
  },
  link: {
    fontFamily: "'Inter', sans-serif",
    textDecoration: "none",
    color: "#1a1a1a",           // Slightly off-black for a more modern look
    fontWeight: "700",         // This makes it Bold
    fontSize: "1rem",       // Slightly smaller font with bolding looks cleaner
    letterSpacing: "-0.2px",   // Tighter tracking for a high-end "tech" feel
    transition: "all 0.3s ease",
    marginLeft: "1.2rem",      // Increased spacing between links for better readability
    display: "inline-block",    // Necessary for transform animations to work
    cursor: "pointer",
  },
  button: {
    backgroundColor: "#16a34a",
    color: "white",
    border: "none",
    padding: "0.6rem 1.2rem",
    borderRadius: "50px",
    fontWeight: "600",
    cursor: "pointer",
    boxShadow: "0 4px 10px rgba(22, 163, 74, 0.2)",
  }
};

export default Navbar;