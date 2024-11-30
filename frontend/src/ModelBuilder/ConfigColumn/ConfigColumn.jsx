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

  const handleInputChange = (field, value) => {
    setBlockInputs(prevInputs => ({
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
    setBlockCount(prevCount => prevCount + 1);

    const leftTrapezoidBase = blockInputs.inputSize / 10;
    const rightTrapezoidBase = blockInputs.outputSize / 10;
    const middleRectangleHeight = blockInputs.hiddenSize / 10;
    const middleRectangleWidth = blockInputs.numHiddenLayers * 10;

    const newLayer = {
      id: `block-${newBlockId}`,
      name: `Block ${newBlockId + 1}`,
      position: { x: 0, y: 0 },
      leftTrapezoid: { base: leftTrapezoidBase, height: middleRectangleHeight },
      rightTrapezoid: { base: rightTrapezoidBase, height: middleRectangleHeight },
      middleRectangle: { width: middleRectangleWidth, height: middleRectangleHeight },
    };

    createLayers([newLayer]);

    // Reset inputs after adding a block
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