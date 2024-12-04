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

const TrainingGraph = ({ graphData }) => {
  const data = {
    labels: graphData.epochs, // X-axis labels (epochs)
    datasets: [
      {
        label: "Loss",
        data: graphData.losses, // Y-axis values for Loss
        borderColor: "rgba(255, 99, 132, 1)",
        backgroundColor: "rgba(255, 99, 132, 0.2)",
        borderWidth: 2,
        fill: true,
      },
      {
        label: "Accuracy",
        data: graphData.accuracies, // Y-axis values for Accuracy
        borderColor: "rgba(54, 162, 235, 1)",
        backgroundColor: "rgba(54, 162, 235, 0.2)",
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
        text: "Training Progress",
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
          text: "Value",
        },
        min: 0, // Set the minimum value for the y-axis
        max: 1.5, // Adjust the maximum value as needed (e.g., 1.5 for metrics like accuracy or loss)
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