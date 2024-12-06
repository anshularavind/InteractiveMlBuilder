// GraphParser.js
import React from "react";
import TrainingGraph from "./TrainingGraph";

const GraphParser = ({ backendResults, selectedDataset }) => {
  let graphData = {
    epochs: [],
    losses: [],
    accuracies: [],
  };
  let accuracyMin = 0;
  let accuracyMax = 1 / 1.1;
  let lossMin = 0;
  let lossMax = 1 / 1.1;

  const isClassification = selectedDataset === "MNIST" || selectedDataset === "CIFAR10";

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
    lossMin = Math.min(...losses);
    lossMax = Math.max(...losses);

    // Parse the output string for accuracies
    const outputString = backendResults.output;
    const outputLines = outputString.trim().split("\n");

    const accuracies = [];
    outputLines.forEach((line) => {
      let accuracyMatch;
      if (isClassification) {
        accuracyMatch = line.match(/(Epoch #(\d+)|Final) accuracy(.*): ([\d.]+)/);
      } else {
        accuracyMatch = line.match(/(Epoch #(\d+)|Final) MSE(.*): ([\d.]+)/);
      }
      if (accuracyMatch) {
        const accuracy = parseFloat(accuracyMatch[accuracyMatch.length - 1]);
        accuracies.push(accuracy);
      }
    });

    accuracyMin = Math.min(...accuracies);
    accuracyMax = Math.max(...accuracies);

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
            yMin={lossMin / 1.1}
            yMax={lossMax * 1.1} // Adjust y-axis max
            lineColor="rgba(255, 99, 132, 1)" // Red color for loss
          />
          <TrainingGraph
            graphData={accuracyGraphData}
            title={isClassification ? "Training Accuracy" : "Training MSE"}
            yLabel={isClassification ? "Accuracy %" : "MSE"}
            yMin={accuracyMin / 1.1}
            yMax={accuracyMax} // Since accuracy is between 0 and 1
            lineColor="rgba(54, 162, 235, 1)" // Blue color for accuracy
          />
        </div>
      )}
    </div>
  );
};

export default GraphParser;