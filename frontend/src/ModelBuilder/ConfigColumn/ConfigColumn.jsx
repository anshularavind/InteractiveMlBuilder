import React, { useState } from "react";
import DatasetSelection from "./Components/DatasetSelection";
import LayerSelection from "./Components/LayerSelection";
import ModelConfig from "./Components/ModelConfig";
import Train from "./Components/Train";
import "../ModelBuilder.css";

function ConfigColumn({
  selectedDataset,
  datasetDropdownOpen,
  toggleDatasetDropdown,
  datasetItems,
  handleDatasetClick,
  layerItems,
  createLayers,
}) {
  const [blockInputs, setBlockInputs] = useState({
    inputSize: 0,
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 0,
  });
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);
  const [blockCount, setBlockCount] = useState(0);

  const SCALING_CONSTANT = 25; // Scaling constant for visual representation
  const log_base = Math.log(1.5); // Base of logarithm for scaling

  const handleInputChange = (field, value) => {
    setBlockInputs((prevInputs) => ({
      ...prevInputs,
      [field]: Math.max(0, parseInt(value) || 0),
    }));
  };

  const handleLayerClick = (item) => {
    setSelectedLayer(item.value);
    setLayerDropdownOpen(false);
  };

  const addBlock = () => {
    const newBlockId = blockCount;
    setBlockCount((prevCount) => prevCount + 1);

    // Calculate precomputed sizes and apply ln and scaling
    const inputSize = Math.log(blockInputs.inputSize || 1) * SCALING_CONSTANT / log_base;
    const outputSize = Math.log(blockInputs.outputSize || 1) * SCALING_CONSTANT / log_base;
    const hiddenSize = Math.log(blockInputs.hiddenSize || 1) * SCALING_CONSTANT / log_base;
    const numHiddenLayers = blockInputs.numHiddenLayers * SCALING_CONSTANT;
    const trapHeight = SCALING_CONSTANT * 2;

    const newLayer = {
      id: `block-${newBlockId}`,
      name: `Block ${newBlockId + 1}`,
      leftTrapezoid: { base: inputSize, height: trapHeight },
      rightTrapezoid: { base: outputSize, height: trapHeight },
      middleRectangle: { width: numHiddenLayers, height: hiddenSize },
    };

    // Pass the new layer to Visualizer via createLayers
    createLayers([newLayer]);

    // Reset inputs
    setBlockInputs({
      inputSize: 0,
      outputSize: 0,
      hiddenSize: 0,
      numHiddenLayers: 0,
    });
    setSelectedLayer(null);
  };

  return (
    <div className="inputBlock">
      <div className="inputBlockHeader">
        <h1><b>Configuration</b></h1>
        <h2><u>Select Dataset</u></h2>
        <DatasetSelection
          selectedItem={selectedDataset}
          dropdownOpen={datasetDropdownOpen}
          toggleDropdown={toggleDatasetDropdown}
          datasetItems={datasetItems}
          handleItemClick={handleDatasetClick}
        />
        <h4>Input Size:</h4>
        <h4>Output Size:</h4>
      </div>
      <div className="inputBlockContent">
        <h2><u>Add Blocks</u></h2>
        <LayerSelection
          selectedItem={selectedLayer}
          dropdownOpen={layerDropdownOpen}
          toggleDropdown={() => setLayerDropdownOpen(!layerDropdownOpen)}
          layerItems={layerItems}
          handleItemClick={handleLayerClick}
        />
        {selectedLayer === "FcNN" && (
          <ModelConfig
            blockInputs={blockInputs}
            handleInputChange={handleInputChange}
            createLayers={addBlock}
          />
        )}
      </div>

      <Train />
    </div>
  );
}

export default ConfigColumn;
