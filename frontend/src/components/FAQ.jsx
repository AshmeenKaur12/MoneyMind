import React, { useState } from 'react';
import './FAQ.css'; // Ensure the path matches where you saved the CSS
import bgPattern from "../assets/11781753-c22b-4b23-879f-e54e998e771c.jpg";
const faqData = [
    {
        question: "What is the Intelligent Agriculture Monitoring and Management System?",
        answer: "This is a digital health-check tool for crops. By analyzing leaf images, our system identifies specific diseases using advanced Image Processing and Machine Learning. It helps farmers detect issues early, reducing crop loss and pesticide overuse."
    },
    {
        question: "What is the goal of this system?",
        answer: "This platform is designed to help farmers and researchers identify plant diseases instantly. By combining the vast PlantVillage dataset with real-world PlantDoc imagery, our model recognizes 38 different classes of healthy and diseased leaves across multiple crop species."
    },
    {
        question: "What crops and diseases are supported?",
        answer: (
            <div className="supported-crops-list">
                <p>Our system recognizes 38 classes across the following crops:</p>
                <ul>
                    <li><strong>Apple:</strong> Scab, Black Rot, Cedar Rust, Healthy</li>
                    <li><strong>Blueberry & Cherry:</strong> Healthy, Powdery Mildew</li>
                    <li><strong>Corn:</strong> Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy</li>
                    <li><strong>Grape:</strong> Black Rot, Esca (Black Measles), Leaf Blight, Healthy</li>
                    <li><strong>Orange:</strong> Haunglongbing (Citrus Greening)</li>
                    <li><strong>Peach:</strong> Bacterial Spot, Healthy</li>
                    <li><strong>Bell Pepper:</strong> Bacterial Spot, Healthy</li>
                    <li><strong>Potato:</strong> Early Blight, Late Blight, Healthy</li>
                    <li><strong>Raspberry, Squash & Strawberry:</strong> Healthy, Powdery Mildew, Leaf Scorch</li>
                    <li><strong>Tomato:</strong> Bacterial Spot, Early/Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy</li>
                </ul>
            </div>
        )
    },
    {
        question: "How should I take the photo for the best results?",
        answer: "For maximum accuracy, ensure the leaf is centered, well-lit by natural light, and placed against a neutral background. Avoid blurry or low-light images."
    },
    {
        question: "How do I upload leaf images for analysis?",
        answer: "Upload a photo of your plant’s leaf, and our AI scans and analyzes it. Get an instant, detailed report showing whether your plant is healthy or stressed."
    },
    {
        question: "Does the background of the photo matter?",
        answer: "Yes. For the best results, place the leaf against a plain, neutral background (like a white sheet or palm) to help the model focus entirely on the leaf's features."
    },
    {
        question: "Can I scan multiple leaves at once?",
        answer: "No, to maintain high diagnostic precision, we recommend scanning one leaf at a time. This allows the model to analyze specific spot patterns and textures clearly."
    }
];

const FAQ = () => {
    const [openIndex, setOpenIndex] = useState(null);
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
    const toggleFAQ = (index) => {
        setOpenIndex(openIndex === index ? null : index);
    };

    return (
        <section className="faq-page" id="faq" style={sectionStyle} // Apply the style here
        >
            <div className="faq-container" >

                <header className="faq-header-section">
                    <div className="leaf-icon">🌿</div>
                    <h1>Frequently Asked Questions</h1>
                    <p>Common questions about our Intelligent Agriculture Monitoring and Management System</p>
                </header>

                <div className="faq-list">
                    {faqData.map((item, index) => (
                        <div
                            key={index}
                            className={`faq-card ${openIndex === index ? 'active' : ''}`}
                            onClick={() => toggleFAQ(index)}
                        >
                            <div className="faq-question">
                                <span>{item.question}</span>
                                <span className="faq-toggle-icon">
                                    {openIndex === index ? '−' : '+'}
                                </span>
                            </div>

                            <div className={`faq-answer ${openIndex === index ? 'show' : ''}`}>
                                <div className="answer-content">
                                    {item.answer}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>

                <footer className="faq-footer">
                    <p>Project Group G3 | Ashmeen, Aditi, Jeeya</p>
                </footer>
            </div>
        </section>
    );
};

export default FAQ;