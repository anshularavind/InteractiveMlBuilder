// GraphParser.js
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

    const epochs = [];
    const losses = [];

    lossLines.forEach((line) => {
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

  // Prepare data for the loss graph
  const lossGraphData = {
    epochs: graphData.epochs,
    values: graphData.losses,
  };

  // Prepare data for the accuracy graph
  const accuracyGraphData = {
    epochs: graphData.epochs,
    values: graphData.accuracies,
  };

  return (
    <div>
      {backendResults && (
        <div>
          <TrainingGraph
            graphData={lossGraphData}
            title="Training Loss"
            yLabel="Loss"
            yMin={0}
            yMax={Math.max(...graphData.losses) * 1.1} // Adjust y-axis max
            lineColor="rgba(255, 99, 132, 1)" // Red color for loss
          />
          <TrainingGraph
            graphData={accuracyGraphData}
            title="Training Accuracy"
            yLabel="Accuracy"
            yMin={0}
            yMax={1} // Since accuracy is between 0 and 1
            lineColor="rgba(54, 162, 235, 1)" // Blue color for accuracy
          />
        </div>
      )}
    </div>
  );
};

export default GraphParser;