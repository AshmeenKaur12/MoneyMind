import React from "react";
import bgPattern from "../assets/11781753-c22b-4b23-879f-e54e998e771c.jpg";
import plantImg from "../assets/0d59838b-681a-4750-a707-23b09fec7916.png";
const HomepageSection = () => {
    return (
        <section style={styles.section} id="home">
            {/* Background Image Layer - Increased Opacity */}
            <div
                style={{
                    ...styles.bgOverlay,
                    backgroundImage: `linear-gradient(rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.1)), url(${bgPattern})`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            backgroundAttachment: 'fixed',
                            width: '100%',
                            minHeight: '100vh',
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center'
                }}
            />

            {/* Decorative Rings - Slightly darker for contrast */}
            <div style={styles.ringLarge}></div>
            <div style={styles.ringSmall}></div>

            <div style={styles.container}>
                {/* Left side: Plant Visuals */}
                <div style={styles.left}>
                    <div style={styles.ovalFrame}>
                        <img
                            src={plantImg}
                            style={styles.plantImage}
                        />


                    </div>
                </div>

                {/* Right side: Branding & CTA */}
                <div style={styles.right}>
                    <h1 style={styles.headline}>
                        Intelligent  <span style={styles.highlight}>Agriculture Monitoring</span> and <span style={styles.brandName}>Management System</span>
                    </h1>
                    <p style={styles.description}>
                        Monitor your crops and detect diseases instantly using AI-powered diagnostics.
                        Our system analyzes plant health issues in real-time to help farmers
                        ensure a sustainable and healthy yield.
                    </p>

                </div>
            </div>
        </section>
    );
};

const styles = {
    section: {
        padding: "6rem 2rem",
        backgroundColor: "#ffffff",
        fontFamily: "'Inter', sans-serif",
        position: "relative",
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        minHeight: "90vh",
    },
    bgOverlay: {
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        backgroundSize: "cover",
        backgroundPosition: "center",
        // CHANGED: Increased from 0.1 to 0.4 for better visibility
        opacity: 0.4,
        zIndex: 1,
        pointerEvents: "none",
    },
    ringLarge: {
        position: "absolute",
        width: "200px",
        height: "200px",
        borderRadius: "50%",
        // CHANGED: Slightly darker green to stand out against the background
        border: "30px solid #d4e7da",
        left: "-40px",
        top: "30%",
        zIndex: 2,
    },
    ringSmall: {
        position: "absolute",
        width: "60px",
        height: "60px",
        borderRadius: "50%",
        border: "12px solid #d4e7da",
        left: "35%",
        top: "10%",
        zIndex: 2,
    },
    container: {
        maxWidth: "1100px",
        margin: "0 auto",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        width: "100%",
        zIndex: 10,
    },
    left: {
        flex: "1",
        position: "relative",
        display: "flex",
        justifyContent: "center",
    },
    // ... inside your styles object

    ovalFrame: {
        width: "320px",
        height: "540px",
        borderRadius: "160px",

        position: "relative",
        // We use a semi-transparent background to let the pattern show slightly

        backdropFilter: "blur(5px)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        /* CRITICAL: Allows leaves to overlap the border */
        overflow: "visible",
    },

    plantImage: {
        /* Set width to 170% as requested */
        width: "310%",
        height: "auto",
        position: "absolute",
        /* Center it: At 170% width, -35% left perfectly centers the image */
        left: "-125%",
        /* Sit the pot slightly above the bottom curve of the oval */
        bottom: "2%",
        zIndex: 3,
        /* Ensures no background color is added */
        backgroundColor: "transparent",
        /* Object-fit contain prevents stretching */
        objectFit: "contain",
        /* Optional: subtle drop shadow to make the plant pop from the frame */
        filter: "drop-shadow(0 10px 20px rgba(0,0,0,0.08))",
    },
    right: {
        marginTop: '0.1rem',
        flex: "5",
        paddingLeft: "4rem",
    },
    headline: {
        fontSize: "3.5rem",
        fontWeight: "800",
        color: "#2d4a31",
        lineHeight: "1.1",
        marginBottom: "1.5rem",
        // Added a very subtle text shadow to lift text off the background
        textShadow: "0 2px 4px rgba(255,255,255,0.5)",
    },
    highlight: {
        color: "#77c393",
    },
    description: {
        fontSize: "1.2rem",
        color: "#4a5a4d", // Slightly darker green-grey
        lineHeight: "1.6",
        marginBottom: "2.5rem",
    },
    
};

export default HomepageSection;