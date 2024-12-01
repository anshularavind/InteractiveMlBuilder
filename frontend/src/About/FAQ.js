import React, { useState } from "react";
import "./FAQ.css"; // Ensure you create a CSS file for styling the FAQ component

function FAQ() {
  const [openQuestion, setOpenQuestion] = useState(null);

  const toggleQuestion = (index) => {
    setOpenQuestion(openQuestion === index ? null : index);
  };

  const faqs = [
    {
      question: "What is this app about?",
      answer: "This app is designed to make building machine learning models simple and accessible for everyone.",
    },
    {
      question: "How do I get started?",
      answer: "Navigate to the 'Model Builder' section, select a dataset, and start adding blocks to configure your model. The training section below the model types will show inisghts on specific metrics, visualizations, and outputs tailored to these architectures. ",
    },
    {
      question: "What datasets are supported?",
      answer: "You can use preloaded datasets like MNIST and CIFAR-10.",
    },
    {
      question: "Can I save my model?",
      answer: "Yes, you can export your model in various formats, such as JSON or ONNX, for deployment or further training.",
    },
    {
      question: "Is this app free to use?",
      answer: "Yes, this app is free to use for educational and non-commercial purposes.",
    },
  ];

  return (
    <div className="faq">
      <h2>Frequently Asked Questions</h2>
      <div className="faq-list">
        {faqs.map((faq, index) => (
          <div key={index} className="faq-item">
            <div
              className="faq-question"
              onClick={() => toggleQuestion(index)}
            >
              {faq.question}
              <span>{openQuestion === index ? "-" : "+"}</span>
            </div>
            {openQuestion === index && (
              <div className="faq-answer">{faq.answer}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default FAQ;
