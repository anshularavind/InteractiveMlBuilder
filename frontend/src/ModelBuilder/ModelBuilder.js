import React, { useState } from "react";
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css";

function ModelBuilder(getAccessTokenSilently) {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetDropdownOpen, setDatasetDropdownOpen] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);

  const [layers, setLayers] = useState([]);
  const [blockInputs, setBlockInputs] = useState({
    inputSize: 0,
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 0,
  });

  // Dataset and layer options
  const datasetItems = [
    { value: "MNIST", label: "MNIST" },
    { value: "CIFAR 10", label: "CIFAR 10" },
  ];

  const layerItems = [
    { value: "FcNN", label: "FcNN" },
    { value: "Conv", label: "Conv" },
  ];

  // Handlers for dataset and layer dropdowns
  const handleDatasetClick = (item) => {
    setSelectedDataset(item.value);
    setDatasetDropdownOpen(false);
  };

  const toggleDatasetDropdown = () => {
    setDatasetDropdownOpen(!datasetDropdownOpen);
  };

  const handleLayerClick = (item) => {
    setSelectedLayer(item.value);
    setLayerDropdownOpen(false);
  };

  const toggleLayerDropdown = () => {
    setLayerDropdownOpen(!layerDropdownOpen);
  };

  // Input change handler for block configurations
  const handleInputChange = (field, value) => {
    const sanitizedValue = Math.max(0, parseInt(value) || 0);
    console.log(`Updating ${field}:`, sanitizedValue);
    setBlockInputs({ ...blockInputs, [field]: sanitizedValue });
  };

  // Generate JSON for model configuration
  const generateJson = (updatedLayers) => {
    console.log("Current Block Inputs:", blockInputs);
    console.log("Current Layers:", layers);
    console.log("Selected Dataset:", selectedDataset);

    const username = "test_user5"; // Replace with dynamic username if needed

    const modelBuilderJson = {
      username, // Ensure username is included
      model_config: {
        input: blockInputs.inputSize,
        output: blockInputs.outputSize,
        dataset: selectedDataset,
        LR: blockInputs.learningRate?.toString() || "0.001",
        batch_size: blockInputs.batchSize,
        blocks: updatedLayers.map((layer) => ({
          block: layer.type || "FcNN",
          params: {
            output_size: layer.params.output_size,
            hidden_size: layer.params.hidden_size,
            num_hidden_layers: layer.params.num_hidden_layers,
          },
        })),
      },
      dataset: selectedDataset, // Include dataset at root level as well
    };

    console.log("Generated JSON:", modelBuilderJson); // Log the JSON
    return modelBuilderJson; // Return the JSON
  };

  // Function to send JSON to the backend
  const sendJsonToBackend = async (json) => {
    try {
      const response = await fetch("http://127.0.0.1:4000/api/define-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${getAccessTokenSilently}`,
        },
        body: JSON.stringify(json),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log("Response from backend:", data);
    } catch (error) {
      console.error("Error sending JSON to backend:", error);
    }
  };

  // Function to add layers
  const createLayers = (newLayers) => {
    setLayers((prevLayers) => {
      const updatedLayers = [...prevLayers, ...newLayers];
      generateJson(updatedLayers);
      return updatedLayers;
    });
  };

  // Function to handle layer dragging
  const onLayerDragStop = (id, data) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === id
          ? { ...layer, position: { x: data.x, y: data.y } }
          : layer
      )
    );
  };

  
  const removeLastBlock = () => {
    setLayers((prevLayers) => {
      if (prevLayers.length === 0) return prevLayers; // No blocks to remove
      return prevLayers.slice(0, -1); // Removes the last block
    });
  };

  return (
    <div className="container">
      <ConfigColumn
        selectedDataset={selectedDataset}
        datasetDropdownOpen={datasetDropdownOpen}
        toggleDatasetDropdown={toggleDatasetDropdown}
        datasetItems={datasetItems}
        handleDatasetClick={handleDatasetClick}
        selectedLayer={selectedLayer}
        layerDropdownOpen={layerDropdownOpen}
        toggleLayerDropdown={toggleLayerDropdown}
        layerItems={layerItems}
        handleLayerClick={handleLayerClick}
        blockInputs={blockInputs}
        handleInputChange={handleInputChange}
        createLayers={createLayers}
        removeLastBlock={removeLastBlock} // Pass removeLastBlock to ConfigColumn
      />
      <Visualizer layers={layers} onLayerDragStop={onLayerDragStop} />
      <button
        className="sendBackend"
        onClick={() => sendJsonToBackend(generateJson(layers))}
      >
        Send JSON
      </button>
    </div>
  );
}

export default ModelBuilder;

