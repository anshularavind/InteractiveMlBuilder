import React, { useState, useEffect } from "react";
import DatasetSelection from "./Components/DatasetSelection";
import LayerSelection from "./Components/LayerSelection";
import FcNNModelConfig from "./Components/FcNNModelConfig";
import ConvModelConfig from "./Components/ConvModelConfig";
import Train from "./Components/Train";
import "../ModelBuilder.css";

function ConfigColumn({
  selectedDataset,
  setSelectedDataset,
  datasetSizesMap,
  datasetDropdownOpen,
  toggleDatasetDropdown,
  datasetItems,
  handleDatasetClick,
  createLayers,
  removeLastBlock,
  layers,
  trainInputs,
  setTrainInputs,
  loadModelConfig,
}) {
  const [blockInputs, setBlockInputs] = useState({
    outputSize: 1,
    hiddenSize: 0,
    numHiddenLayers: 0,
    kernelSize: 3,
    numKernels: 1,
  });

  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);

  const [datasetSizes, setDatasetSizes] = useState({ inputSize: 0, outputSize: 0 });

  useEffect(() => {
    setDatasetSizes(datasetSizesMap[selectedDataset] ?? { inputSize: 0, outputSize: 0 });
  }, [selectedDataset]);
  let newBlock = null;
  const addBlock = () => {
    const SCALING_CONSTANT = 25;
    const log_base = Math.log(1.5);
    const log_base_pool = Math.log(3);

    if (newBlock === null) {
      let inputSize = datasetSizes.inputSize;
      if (layers.length > 0) {
        if (layers[layers.length - 1].type === "Conv" && selectedLayer === "FcNN") {
          inputSize = layers[layers.length - 1].params.output_size ** 2 * layers[layers.length - 1].params.num_kernels;
        } else {
            inputSize = layers[layers.length - 1].params.output_size;
        }
      }

      newBlock = {
        id: layers.length,
        chosenLayer: selectedLayer,
        datasetInputSize: datasetSizes.inputSize,
        prevOutputDim: layers.length > 0 ? layers[layers.length - 1].params.output_size : datasetSizes.inputSize,
        inputSize: inputSize,
        outputSize: blockInputs.outputSize,
        numHiddenLayers: blockInputs.numHiddenLayers,
        hiddenSize: blockInputs.hiddenSize,
        layersLength: layers.length,
        kernelSize: blockInputs.kernelSize,
        numKernels: blockInputs.numKernels,
      };
    }

    const inputSizeScaled =
      (Math.log((newBlock.inputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
    const outputSizeScaled =
      (Math.log((newBlock.outputSize + 1) || 1) * SCALING_CONSTANT) / log_base;

    let newLayer;
    if (newBlock.chosenLayer === "Conv") {
      let previousOutputDim;
      if (newBlock.layersLength === 0) {
        previousOutputDim = newBlock.datasetInputSize;
        if (selectedDataset === "MNIST" || selectedDataset === "CIFAR10") {
          previousOutputDim = Math.sqrt(previousOutputDim);
        }
      } else {
        previousOutputDim = newBlock.prevOutputDim;
      }

      const outputSize= newBlock.outputSize;
      const poolSize = newBlock.outputSize;
      const poolSizeScaled = (Math.log((poolSize + 1) || 1) * SCALING_CONSTANT) / log_base_pool;
      const kernelSize = newBlock.kernelSize;
      const kernelSizeScaled = (Math.log((kernelSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      const numKernels = newBlock.numKernels;
      const outputLengthScaled = (Math.log((previousOutputDim + 1) || 1) * SCALING_CONSTANT) / log_base_pool;

      newLayer = {
        id: `block-${newBlock.id}`,
        name: `${newBlock.id + 1}`,
        type: "Conv",
        params: {
          input_size: newBlock.inputSize,
          output_size: outputSize,
          kernel_size: kernelSize,
          num_kernels: numKernels,
          stride: 1,
          padding: Math.floor(kernelSize / 2),
        },
        visParams: {
          kernel_size: kernelSizeScaled,
          poolingBlock: { smallBlock: poolSizeScaled, largeBlock: outputLengthScaled },
          width: kernelSizeScaled + numKernels * 5 + outputLengthScaled + 10,
        },
      };
    } else if (newBlock.chosenLayer === "FcNN") {
      let hiddenSizeScaled;
      if (newBlock.numHiddenLayers === 0) {
        hiddenSizeScaled = (inputSizeScaled + outputSizeScaled) / 2;
      } else {
        hiddenSizeScaled = (Math.log((newBlock.hiddenSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      }
      const numHiddenLayersScaled = newBlock.numHiddenLayers * SCALING_CONSTANT;
      const trapHeight = SCALING_CONSTANT * 2;

      newLayer = {
        id: `block-${newBlock.id}`,
        name: `${newBlock.id + 1}`,
        type: "FcNN",
        params: {
          input_size: newBlock.inputSize,
          output_size: newBlock.outputSize,
          hidden_size: newBlock.hiddenSize,
          num_hidden_layers: newBlock.numHiddenLayers,
        },
        visParams: {
          leftTrapezoid: { base: inputSizeScaled, height: trapHeight },
          rightTrapezoid: { base: outputSizeScaled, height: trapHeight },
          middleRectangle: {
            width: numHiddenLayersScaled,
            height: hiddenSizeScaled,
          },
          width: trapHeight + numHiddenLayersScaled + trapHeight,
        },
      };
    }

    createLayers([newLayer]);

    setBlockInputs({
      outputSize: 1,
      hiddenSize: 0,
      numHiddenLayers: 0,
      kernelSize: 3,
      numKernels: 1,
    });
    setSelectedLayer(null);
    newBlock = null;
  };

  // Handle JSON configuration (if any)
  const handleJsonConfig = (jsonConfig) => {
    setSelectedDataset(jsonConfig.dataset);
    setTrainInputs({
      lr: jsonConfig.LR,
      batch_size: jsonConfig.batch_size,
      epochs: jsonConfig.epochs,
    });

    setDatasetSizes({
      inputSize: jsonConfig.input,
      outputSize: jsonConfig.output,
    });

    let previousOutputDim = jsonConfig.input;
    switch (jsonConfig.dataset) {
        case "CIFAR10":
          previousOutputDim /= 3;
        case "MNIST":
          previousOutputDim = Math.sqrt(previousOutputDim);
          break;
    }
    let index = 0
    let previousNumKernels = jsonConfig.dataset === "CIFAR10" ? 3 : 1;
    jsonConfig.blocks.forEach((block) => {
      if (block.block === "Pool")
        return;
      let outputSize = block.params.output_size;
      if (block.block === "Conv") {
        outputSize = jsonConfig.dataset === "MNIST" || jsonConfig.dataset === "CIFAR10"
          ? Math.sqrt(block.params.output_size / block.params.num_kernels)
          : block.params.output_size / block.params.num_kernels;
        previousNumKernels = block.params.num_kernels;
      }
      let inputSize = previousOutputDim;
      if (block.block === "FcNN" && previousNumKernels !== -1) {
        inputSize = jsonConfig.dataset === "MNIST" || jsonConfig.dataset === "CIFAR10"
            ? previousOutputDim ** 2
            : previousOutputDim;
        inputSize = inputSize * previousNumKernels;
        previousNumKernels = -1;
      }

      newBlock = {
        id: index,
        chosenLayer: block.block,
        datasetInputSize: jsonConfig.input,
        prevOutputDim: previousOutputDim,
        inputSize: inputSize,
        outputSize: outputSize,
        numHiddenLayers: block.params.num_hidden_layers,
        hiddenSize: block.params.hidden_size,
        layersLength: -1,
        kernelSize: block.params.kernel_size,
        numKernels: block.params.num_kernels,
      };
      addBlock();
      previousOutputDim = outputSize;
      index++;
    });
  };

  const handleBlockInputChange = (field, value) => {
    setBlockInputs((prevInputs) => {
      let newValue;
      if (field === "numHiddenLayers") {
        newValue = Math.max(0, parseInt(value) || 0);
      } else {
        newValue = Math.max(1, parseInt(value) || 0);
      }
      return {
        ...prevInputs,
        [field]: newValue,
      };
    });
  };

  const handleTrainingInputChange = (field, value) => {
    setTrainInputs((prevInputs) => {
      let newValue;
      switch (field) {
        case "lr":
          newValue = parseFloat(value) || 0;
          newValue = Math.max(0, Math.min(0.1, newValue));
          break;
        case "batch_size":
          newValue = parseInt(value) || 0;
          newValue = Math.max(1, Math.min(2048, newValue));
          break;
        case "epochs":
          newValue = parseInt(value) || 0;
          newValue = Math.max(1, Math.min(100, newValue));
          break;
        default:
          newValue = value;
      }
      return {
        ...prevInputs,
        [field]: newValue,
      };
    });
  };

  const handleLayerClick = (item) => {
    setSelectedLayer(item.value);
    setLayerDropdownOpen(false);
  };

  useEffect(() => {
    let cachedModelConfig = loadModelConfig();
    sessionStorage.setItem("cachedModelConfig", "{}"); // Clear cache to avoid duplicate storing
    if (
      cachedModelConfig !== null &&
      layers.length === 0 &&
      cachedModelConfig.blocks.length > 0
    ) {
      handleJsonConfig(cachedModelConfig);
    }
  }, [layers, loadModelConfig]);

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
         <div className="datasetInfo">
    <div className="infoItem">
      <span className="infoLabel">Input Size</span>
      <span className="infoValue">{datasetSizes.inputSize}</span>
    </div>
    <div className="infoItem">
      <span className="infoLabel">Output Size</span>
      <span className="infoValue">{datasetSizes.outputSize}</span>
    </div>
  </div>
</div>
      {selectedDataset && (
        <div className="inputBlockContent">
          <h2>Add Blocks</h2>
          <LayerSelection
            selectedItem={selectedLayer}
            dropdownOpen={layerDropdownOpen}
            toggleDropdown={() => setLayerDropdownOpen(!layerDropdownOpen)}
            handleItemClick={handleLayerClick}
            layers={layers}
          />
          {selectedLayer === "FcNN" && (
            <FcNNModelConfig
              blockInputs={blockInputs}
              handleInputChange={handleBlockInputChange}
              createLayers={addBlock}
            />
          )}
          {selectedLayer === "Conv" && (
            <ConvModelConfig
              blockInputs={blockInputs}
              handleInputChange={handleBlockInputChange}
              createLayers={addBlock}
              selectedDataset={selectedDataset}
            />
          )}
          <br />
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
          <br />
          <Train
            trainInputs={trainInputs}
            handleInputChange={handleTrainingInputChange}
          />
        </div>
      )}
    </div>
  );
}

export default ConfigColumn;
