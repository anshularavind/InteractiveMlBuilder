import React, { useState } from "react";
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css";

function ModelBuilder() {
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
    setBlockInputs({ ...blockInputs, [field]: sanitizedValue });
  };

  const generateJson = (updatedLayers) => {
    const username = "test_user5";
  
    const modelBuilderJson = {
      username,
      model_config: {
        input: blockInputs.inputSize,
        output: blockInputs.outputSize,
        dataset: selectedDataset,
        lr: blockInputs.learningRate || 0.001,
        batch_size: blockInputs.batchSize || 32,
        blocks: updatedLayers.map((layer) => ({
          block: layer.name || "FcNN",
          params: layer.params || { output_size: 0, hidden_size: 0, num_hidden_layers: 0 }, // Default params
        })),
      },
      dataset: selectedDataset,
    };
  
    console.log("Generated JSON:", modelBuilderJson);
    return modelBuilderJson;
  };
  

  const createLayers = (newLayers) => {
    setLayers((prevLayers) => {
      const updatedLayers = [...prevLayers, ...newLayers];
      generateJson(updatedLayers); // Generate JSON whenever a new block is added
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
        <Visualizer layers={layers} onLayerDragStop={onLayerDragStop} />
      </div>
    </div>
  );
}

export default ModelBuilder;
