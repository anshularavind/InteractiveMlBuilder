import React, { useState } from "react";
import { useAuth0 } from "@auth0/auth0-react"; // Import useAuth0
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css";

function ModelBuilder() {
  const { getAccessTokenSilently, loginWithRedirect, logout, user, isAuthenticated, isLoading } = useAuth0(); // Destructure Auth0 methods
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

  const datasetItems = [
    { value: "MNIST", label: "MNIST" },
    { value: "CIFAR 10", label: "CIFAR 10" },
  ];

  const layerItems = [
    { value: "FcNN", label: "FcNN" },
    { value: "Conv", label: "Conv" },
  ];

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

  const handleInputChange = (field, value) => {
    const sanitizedValue = Math.max(0, parseInt(value) || 0);
    setBlockInputs({ ...blockInputs, [field]: sanitizedValue });
  };

  const generateJson = (updatedLayers) => {
    const modelBuilderJson = {
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
      dataset: selectedDataset,
    };

    return modelBuilderJson;
  };

  const sendJsonToBackend = async (json) => {
    try {
        // Get token before making request
        const token = await getAccessTokenSilently();
        
        const response = await fetch("https://127.0.0.1:4000/api/define-model", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`,
                "Accept": "application/json"
            },
            credentials: "include",
            mode: "cors",
            body: JSON.stringify(json)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error("Error sending JSON to backend:", error);
        throw error; // Re-throw to handle in component
    }
};

  const createLayers = (newLayers) => {
    setLayers((prevLayers) => {
      const updatedLayers = [...prevLayers, ...newLayers];
      generateJson(updatedLayers);
      return updatedLayers;
    });
  };

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
      if (prevLayers.length === 0) return prevLayers;
      return prevLayers.slice(0, -1);
    });
  };

  return (
    <div>
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
        />
        <button className="deleteButton" onClick={removeLastBlock}>
          Remove Last Block
        </button>
        <Visualizer layers={layers} onLayerDragStop={onLayerDragStop} />
        <button
          className="sendBackend"
          onClick={() => sendJsonToBackend(generateJson(layers))}
        >
          Send Json
        </button>
      </div>
    </div>
  );
}

export default ModelBuilder;