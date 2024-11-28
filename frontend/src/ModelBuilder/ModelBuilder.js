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

  const createLayers = () => {
    const newLayers = [];
    for (let i = 0; i < blockInputs.numHiddenLayers; i++) {
      newLayers.push({
        id: layers.length + i,
        name: `Layer ${i + 1}`,
        position: { x: 0, y: i * 60 },
      });
    }
    setLayers([...layers, ...newLayers]);
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

  // Function to remove the last layer
const removeLastLayer = () => {
  setLayers((prevLayers) => prevLayers.slice(0, -1)); // Remove the last layer
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