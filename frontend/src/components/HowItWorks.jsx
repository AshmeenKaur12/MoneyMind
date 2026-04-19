import { useState } from "react";
import "./HowItWorks.css";

export default function HowItWorks() {
  const stepsData = [
    {
  title: "Upload clear images of your plant leaf.",

  img: "https://images.stockcake.com/public/3/9/8/398dd399-bec7-4245-9be0-a2b5642c8fd2/luminous-monstera-leaf-stockcake.jpg"},
    {
      title: "Scan the image to extract relevant features.",
      
      img: "https://images.unsplash.com/photo-1501004318641-b39e6451bec6"
    },
    {
      title: "Analyze using deep learning and machine learning for accurate detection.",
      
      img: "https://images.stockcake.com/public/3/d/3/3d38f126-8e14-4bef-82b5-7b0ba7cdbee0_large/solo-leaf-sky-stockcake.jpg"
    },
    {
      title: "Classify the plant as healthy or unhealthy..",
      
      img: "https://images.stockcake.com/public/d/2/3/d2372b7a-4263-4c5a-80e2-f921a7f2c4e3_large/tropical-leaf-sky-stockcake.jpg"
    },
    {
      title: "Generate a detailed, instant plant health analysis",
      
      img: "https://images.stockcake.com/public/6/2/1/62159c4f-9383-4b60-9c06-1f01211e6a93_large/leaf-illuminated-sunlight-stockcake.jpg"
    }
  ];

  const [activeIndex, setActiveIndex] = useState(0);

  return (
    <div className="how-it-works-page" id="how-it-works">
      <div className="full-width-container">
        
        <div className="visual-section">
          <div className="phone-stack">
            <div className="phone-main">
              <img 
                src={stepsData[activeIndex].img} 
                alt="Plant diagnosis" 
                // This class is the key to making the image full-height
                className={!stepsData[activeIndex].boldText ? "full-image" : ""}
              />
              {stepsData[activeIndex].boldText && (
                <div className="phone-content">
                  <h3>{stepsData[activeIndex].boldText}</h3>
                  <p>{stepsData[activeIndex].desc}</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="content-section">
          <h1 className="main-title">How <span>it works</span></h1>

          <ul className="steps-list">
            {stepsData.map((step, index) => (
              <li
                key={index}
                className={activeIndex === index ? "step-item active" : "step-item"}
                onMouseEnter={() => setActiveIndex(index)}
              >
                <span className="step-number">{index + 1}</span>
                <p className="step-text">{step.title}</p>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}