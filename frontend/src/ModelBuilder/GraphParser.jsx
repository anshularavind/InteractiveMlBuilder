import React from "react";
import TrainingGraph from "./TrainingGraph";

const GraphParser = ({ backendResults }) => {
  let graphData = {
    epochs: [],
    losses: [],
    accuracies: [],
  };

  if (backendResults) {
    // Parse the loss string
    const lossString = backendResults.loss;
    const lossLines = lossString.trim().split("\n");

    // Initialize arrays to hold epochs and losses
    const epochs = [];
    const losses = [];

    lossLines.forEach((line) => {
      // Match "Epoch X/Y, Loss: Z"
      const lossMatch = line.match(/Epoch (\d+)\/\d+, Loss: ([\d.]+)/);
      if (lossMatch) {
        const epoch = parseInt(lossMatch[1], 10);
        const loss = parseFloat(lossMatch[2]);
        epochs.push(epoch);
        losses.push(loss);
      }
    });

    // Parse the output string for accuracies
    const outputString = backendResults.output;
    const outputLines = outputString.trim().split("\n");

    const accuracies = [];

    outputLines.forEach((line) => {
      // Match "Epoch #X accuracy (%): Y"
      const accuracyMatch = line.match(/Epoch #(\d+) accuracy \(%\): ([\d.]+)/);
      if (accuracyMatch) {
        const accuracy = parseFloat(accuracyMatch[2]) / 100; // Convert percentage to decimal
        accuracies.push(accuracy);
      }
    });

    // Build the graphData object
    graphData = {
      epochs,
      losses,
      accuracies,
    };
  }

  return (
    <div>
      {backendResults && <TrainingGraph graphData={graphData} />}
    </div>
  );
};

export default GraphParser;