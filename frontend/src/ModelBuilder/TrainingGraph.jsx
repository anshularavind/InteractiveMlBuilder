// TrainingGraph.js
import React from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const TrainingGraph = ({ graphData, title, yLabel, yMin, yMax, lineColor }) => {
  const data = {
    labels: graphData.epochs, // X-axis labels (epochs)
    datasets: [
      {
        label: yLabel,
        data: graphData.values, // Y-axis values (losses or accuracies)
        borderColor: lineColor,
        backgroundColor: `${lineColor}33`, // Add transparency to the background color
        borderWidth: 2,
        fill: true,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text: title,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: "Epochs",
        },
      },
      y: {
        title: {
          display: true,
          text: yLabel,
        },
        min: yMin,
        max: yMax,
      },
    },
  };

  return (
    <div className="training-graph-container">
      <Line data={data} options={options} />
    </div>
  );
};

export default TrainingGraph;