import React from "react";
import Navbar from "./components/Navbar"; // Adjust path if needed
import HomepageSection from "./components/Homepagesection1";
import FAQ from "./components/FAQ";
import HowItWorks from "./components/HowItWorks";
import About from "./components/About";

function App() {
  return (
    <div>
      <Navbar />
      <HomepageSection />
      <About/>
      <HowItWorks/>
      <FAQ/>
      
    </div>
  );
}

export default App;