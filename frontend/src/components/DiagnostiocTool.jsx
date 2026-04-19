import React from 'react';

const DiagnosticTool = () => {
  return (
    <section id="analysis" style={styles.toolSection}>
      <div style={styles.container}>
        <h2 style={styles.title}>Live Diagnostic Engine</h2>
        <p style={styles.subtitle}>
          Our AI engines (CNN & ML) are ready. Upload your leaf imagery below for real-time analysis.
        </p>
        
        <div style={styles.iframeWrapper}>
          {/* Replace with your local port or deployed Streamlit URL */}
          <iframe
            src="http://localhost:8501/?embedded=true"
            style={styles.iframe}
            title="Streamlit Backend"
            allow="camera"
          ></iframe>
        </div>
      </div>
    </section>
  );
};

const styles = {
  toolSection: { padding: "80px 20px", backgroundColor: "#F7FAF7", textAlign: "center" },
  container: { maxWidth: "1100px", margin: "0 auto" },
  title: { fontSize: "2.5rem", color: "#1b4332", fontWeight: "800", marginBottom: "10px" },
  subtitle: { color: "#636e72", marginBottom: "40px", fontSize: "1.1rem" },
  iframeWrapper: {
    borderRadius: "24px",
    overflow: "hidden",
    boxShadow: "0 20px 50px rgba(0,0,0,0.1)",
    border: "2px solid #E8F0E8",
    background: "#fff"
  },
  iframe: { width: "100%", height: "850px", border: "none" }
};

export default DiagnosticTool;