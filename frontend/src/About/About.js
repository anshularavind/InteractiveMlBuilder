import React from "react";
import FAQ from "./FAQ"; // Import the FAQ component
import "./About.css";

function About() {
  const goBack = () => {
    window.history.back(); // Navigate to the previous page in browser history
  };

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
            The Interactive Machine Learning Model Builder is a web-based application that empowers users to create, customize, and train their 
            own machine learning models without extensive coding knowledge. By providing an interactive and visual platform, users can select from 
            various neural network architectures, adjust hyperparameters, and train models on pre-loaded datasets. The application aims to make machine 
            learning more accessible by simplifying the model-building process and offering real-time training feedback.
          </p>
        </section>

        <section className="about-features">
          <h2>Features</h2>
          <ul>
            <li>Configure fully connected and convolutional neural networks.</li>
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

