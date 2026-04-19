import React from 'react';
import './About.css'; // Ensure your styles are in this file
import bgPattern from "../assets/11781753-c22b-4b23-879f-e54e998e771c.jpg";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

<motion.div
  className="stat-box-large"
  initial={{ opacity: 0, y: 40 }}
  whileInView={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.6 }}
></motion.div>
const Counter = ({ target }) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const duration = 1200;
    const step = target / (duration / 16);

    const interval = setInterval(() => {
      start += step;
      if (start >= target) {
        setCount(target);
        clearInterval(interval);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);

    return () => clearInterval(interval);
  }, [target]);

  return <span>{count.toLocaleString()}</span>;
};
const About = () => {
  // Stats data array for easy management
  const systemStats = [
    { value: "97%", label: "CNN Accuracy" },
    { value: "92%", label: "SVM Accuracy" },
    { value: "85%", label: "Random Forest Accuracy" },
    { value: "61K+", label: "Images Used" },
    { value: "39", label: "Disease Classes" }
  ];
  const sectionStyle = {
    backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.7), rgba(255, 255, 255, 0.7)), url(${bgPattern})`,
    backgroundSize: 'cover',
    backgroundPosition: 'center',
    backgroundAttachment: 'fixed',
    width: '100%',
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center' // Optional: creates a nice parallax effect
  };
  return (
    <div className="blossom-theme">
      {/* ===== HEADER SECTION ===== */}
      <header className="about-header-section">
        <h1>About us</h1>
      </header>

      {/* ===== DESCRIPTION SECTION ===== */}
      <section className="about">
        <div className="about-container">
          <div className="about-text">

            <p className="main-description">
              Plant diseases—caused by fungi, bacteria, and viruses—are the biggest threat to crop quality.
              Leaf diseases are the most common, but identifying them manually is slow and often leads to mistakes.
              For farmers, detecting these issues late means high costs and lost harvests. We created this platform
              to bring Artificial Intelligence directly to the field. By using Deep Learning and Machine Learning,
              our system can automatically scan leaf images and identify diseases in seconds. It acts as an instant
              digital expert, helping farmers take action before the damage spreads.
            </p>
          </div>
        </div>
      </section>

      {/* ===== STATS SECTION ===== */}
      <section className="dashboard-v2" style={sectionStyle}>
        <div className="container">
          <div className="dashboard-header">
            <h3>Technical Capabilities</h3>
            <p>Engineered for speed, scale, and cross-platform reliability.</p>
          </div>

          <div className="pro-stats-grid">
            <div className="stat-box-large">
              <div className="stat-label">Dataset Capacity</div>
              <div className="stat-value">
                <Counter target={63828} />
              </div>
              <p>High-resolution leaf images used for model training and validation.</p>
              <div className="stat-bar-bg"><div className="stat-bar-fill" style={{ width: '90%' }}></div></div>
            </div>

            <div className="stat-box-medium">
              <div className="stat-label">Classification Scope</div>
              <div className="stat-value">39</div>
              <p>Unique disease and health classes identified across 10+ species.</p>
              <div className="tag-group">
                <span>Fungal</span> <span>Viral</span> <span>Bacterial</span>
              </div>
            </div>

            <div className="stat-box-mini-row">
              <div className="mini-box">
                <strong>Real-time</strong>
                <span>Inference Speed</span>
              </div>
              <div className="mini-box">
                <strong>Multi-Model</strong>
                <span>CNN • SVM • RF</span>
              </div>
            </div>
          </div>
        </div>
      </section>


    </div>
  );
};

export default About;