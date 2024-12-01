import React, { useState } from "react";
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css";

function ModelBuilder(getAccessTokenSilently) {
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetDropdownOpen, setDatasetDropdownOpen] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);

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

  const [layers, setLayers] = useState([]);
  const [blockInputs, setBlockInputs] = useState({
    inputSize: 0,
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 0,
  });

  const handleInputChange = (field, value) => {
    const sanitizedValue = Math.max(0, parseInt(value) || 0);
    console.log(`Updating ${field}:`, sanitizedValue);
    setBlockInputs({ ...blockInputs, [field]: sanitizedValue });
  };

  const generateJson = (updatedLayers) => {
    
  console.log("Current Block Inputs:", blockInputs);
  console.log("Current Layers:", layers);
  console.log("Selected Dataset:", selectedDataset);
    
    const username = "test_user5"; // Replace with dynamic username if needed
  
    const modelBuilderJson = {
      username, // Ensure username is included
      model_config: {
        input: blockInputs.inputSize , 
        output: blockInputs.outputSize, 
        dataset: selectedDataset , 
        LR: blockInputs.learningRate?.toString() || "0.001", // Convert LR to string as per example
        batch_size: blockInputs.batchSize , // Default batch size
        blocks: updatedLayers.map((layer) => ({
          block: layer.type || "FcNN", // Default block name
          params: {
            output_size: layer.params.output_size ,
            hidden_size: layer.params.hidden_size ,
            num_hidden_layers: layer.params.num_hidden_layers ,
          },
        })),
      },
      dataset: selectedDataset , // Include dataset at root level as well
    };
  
    console.log("Generated JSON:", modelBuilderJson); // Log the JSON
    return modelBuilderJson; // Return the JSON
  };
  
  const sendJsonToBackend = async (json) => {

    try {
      const response = await fetch("http://127.0.0.1:4000/api/define-model", {

        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${getAccessTokenSilently}`
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
      if (prevLayers.length === 0) return prevLayers; // No blocks to remove
      return prevLayers.slice(0, -1); // Removes the last block
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
