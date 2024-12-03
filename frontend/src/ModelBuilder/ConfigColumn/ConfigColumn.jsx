import React, { useState, useEffect } from "react";
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
  removeLastBlock,
  layers,
}) {
  const [blockInputs, setBlockInputs] = useState({
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 0,
  });
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);
  const [blockCount, setBlockCount] = useState(0);
  const [datasetSizes, setDatasetSizes] = useState({ inputSize: 0, outputSize: 0 });

  useEffect(() => {
    if (selectedDataset === "MNIST") {
      setDatasetSizes({ inputSize: 784, outputSize: 10 });
    } else if (selectedDataset === "CIFAR 10") {
      setDatasetSizes({ inputSize: 3072, outputSize: 10 });
    } else {
      setDatasetSizes({ inputSize: 0, outputSize: 0 });
    }
  }, [selectedDataset]);

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

    let inputSize;
    if (layers && layers.length > 0) {
      inputSize = layers[layers.length - 1].params.output_size;
    } else {
      inputSize = datasetSizes.inputSize;
    }

    const SCALING_CONSTANT = 25;
    const log_base = Math.log(1.5);
    const inputSizeScaled = (Math.log((inputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
    const outputSizeScaled =
      (Math.log((blockInputs.outputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
    const hiddenSizeScaled =
      (Math.log((blockInputs.hiddenSize + 1) || 1) * SCALING_CONSTANT) / log_base;
    const numHiddenLayersScaled = blockInputs.numHiddenLayers * SCALING_CONSTANT;
    const trapHeight = SCALING_CONSTANT * 2;

    const newLayer = {
      id: `block-${newBlockId}`,
      name: `Block ${newBlockId + 1}`,
      type: selectedLayer,
      params: {
        input_size: inputSize,
        output_size: blockInputs.outputSize,
        hidden_size: blockInputs.hiddenSize,
        num_hidden_layers: blockInputs.numHiddenLayers,
      },
      leftTrapezoid: { base: inputSizeScaled, height: trapHeight },
      rightTrapezoid: { base: outputSizeScaled, height: trapHeight },
      middleRectangle: { width: numHiddenLayersScaled, height: hiddenSizeScaled },
    };

    createLayers([newLayer]);

    setBlockInputs({
      outputSize: 0,
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
        <DatasetSelection
          selectedItem={selectedDataset}
          dropdownOpen={datasetDropdownOpen}
          toggleDropdown={toggleDatasetDropdown}
          datasetItems={datasetItems}
          handleItemClick={handleDatasetClick}
        />
        <h4>Input Size: {datasetSizes.inputSize}</h4>
        <h4>Output Size: {datasetSizes.outputSize}</h4>
      </div>
      <div className="inputBlockContent">
        {/* Remove Last Block Button */}
        <div className="remove-block">
          <button
            className="deleteButton"
            onClick={() => {
              console.log("Remove Last Block button clicked");
              removeLastBlock();
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
            handleInputChange={handleInputChange}
            createLayers={addBlock}
          />
        )}
      </div>

      <Train />

      {/* Block Listing Section */}
      <div className="blockList">
        <h2>
          <u>Block Parameters</u>
        </h2>
        {layers && layers.length > 0 ? (
          <ul>
            {layers.map((layer) => (
              <li key={layer.id}>
                <b>{layer.name}:</b>
                <ul>
                  <li>Type: {layer.type}</li>
                  <li>Output Size: {layer.params.output_size}</li>
                  <li>Hidden Size: {layer.params.hidden_size}</li>
                  <li>Number of Hidden Layers: {layer.params.num_hidden_layers}</li>
                </ul>
              </li>
            ))}
          </ul>
        ) : (
          <p>No blocks have been added yet.</p>
        )}
      </div>
    </div>
  );
}

export default ConfigColumn;