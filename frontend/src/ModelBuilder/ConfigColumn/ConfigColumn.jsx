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
  removeLastBlock, // Added here to accept the function as a prop
}) {
  const [blockInputs, setBlockInputs] = useState({
    hiddenSize: 0,
    numHiddenLayers: 0,
  });
  const [independentInputs, setIndependentInputs] = useState({
    inputSize: 0,
    outputSize: 0,
  });
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);
  const [blockCount, setBlockCount] = useState(0);

  const handleBlockInputChange = (field, value) => {
    setBlockInputs((prevInputs) => ({
      ...prevInputs,
      [field]: Math.max(0, parseInt(value) || 0),
    }));
  };

  const handleIndependentInputChange = (field, value) => {
    setIndependentInputs((prevInputs) => ({
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

    const newLayer = {
      id: `block-${newBlockId}`,
      name: `Block ${newBlockId + 1}`,
      type: selectedLayer,
      params: {
        hidden_size: blockInputs.hiddenSize,
        num_hidden_layers: blockInputs.numHiddenLayers,
      },
      leftTrapezoid: { base: 50, height: 50 },
      rightTrapezoid: { base: 50, height: 50 },
      middleRectangle: {
        width: blockInputs.numHiddenLayers * 50,
        height: blockInputs.hiddenSize,
      },
    };

    // Pass the new layer to Visualizer via createLayers
    createLayers([newLayer]);

    // Reset inputs for block-specific fields
    setBlockInputs({
      hiddenSize: 0,
      numHiddenLayers: 0,
    });
    setSelectedLayer(null);
  };

  return (
    <div className="inputBlock">
      <div className="inputBlockHeader">
        <h1>
          <b>Configuration</b>
        </h1>
        <h2>
          <u>Select Dataset</u>
        </h2>
        <DatasetSelection
          selectedItem={selectedDataset}
          dropdownOpen={datasetDropdownOpen}
          toggleDropdown={toggleDatasetDropdown}
          datasetItems={datasetItems}
          handleItemClick={handleDatasetClick}
        />
      </div>
      <div className="sizeInputs">
        <label>
          <h4>Input Size:</h4>
          <input
            type="number"
            value={independentInputs.inputSize}
            onChange={(e) => handleIndependentInputChange("inputSize", e.target.value)}
            className="input-box"
          />
        </label>
        <label>
          <h4>Output Size:</h4>
          <input
            type="number"
            value={independentInputs.outputSize}
            onChange={(e) => handleIndependentInputChange("outputSize", e.target.value)}
            className="input-box"
          />
        </label>
      </div>
      <div className="inputBlockContent">
        {/* Remove Last Block Button */}
        <div className="remove-block">
          <button
            className="deleteButton"
            onClick={() => {
              console.log("Remove Last Block button clicked");
              removeLastBlock(); // Call the passed-in prop function
            }}
          >
            Remove Last Block
          </button>
        </div>
        <h2>
          <u>Add Blocks</u>
        </h2>
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
            handleInputChange={handleBlockInputChange}
            createLayers={addBlock}
          />
        )}
      </div>

      <Train />
    </div>
  );
}

export default ConfigColumn;



