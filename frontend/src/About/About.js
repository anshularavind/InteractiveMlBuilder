import React from "react";
import FAQ from "./FAQ"; // Import the FAQ component
import "./About.css";

function About({ goBack }) {
  return (
    <div className="about-container">
      <div className="about-header">
        <button className="back-button" onClick={goBack}>
          ← Back
        </button>
        <h1>About Our App</h1>
      </div>
      
      <div className="about-content">
        <section className="about-introduction">
          <h2>What is this app?</h2>
          <p>
            This app is designed to empower users to build, configure, and train machine learning models in an intuitive and efficient way. 
            Whether you're a student, researcher, or developer, our tools make prototyping and experimenting with AI simple and accessible.
          </p>
        </section>

        <section className="about-features">
          <h2>Features</h2>
          <ul>
            <li>Upload your own datasets or choose from preloaded ones.</li>
            <li>Configure fully connected, convolutional, and recurrent neural networks.</li>
            <li>Intuitive drag-and-drop model builder interface.</li>
            <li>Export your trained models for further use or deployment.</li>
            <li>Get started with pre-configured templates for common architectures.</li>
          </ul>
        </section>

        <section className="about-faq">
          <FAQ /> {/* Include the FAQ component */}
        </section>
      </div>
    </div>
  );
}

export default About;
