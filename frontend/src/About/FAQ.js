import React, { useState } from "react";
import "./FAQ.css"; // Ensure you create a CSS file for styling the FAQ component

function FAQ() {
  const [openQuestion, setOpenQuestion] = useState(null);

  const toggleQuestion = (index) => {
    setOpenQuestion(openQuestion === index ? null : index);
  };

  const faqs = [
    {
      question: "If there's an FcNN right after a CNN block, why are the CNN's output and the FcNN's input sizes different?",
      answer: "The \"output size\" of the CNN is the side length of the matrix after the conv layer. This is flattened into a 1D array before being passed to the FcNN. The FcNN's input size is the total number of elements in this 1D array. For example, an 8 x 8 image with 16 output channels is flattened into an 8 x 8 x 16 = 1024 element array. For 1D series such as ETTh1, the data is 1D, so a Conv output size of 8 with 16 channels flattens into an 8 x 16 = 128 element array.",
    },
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
