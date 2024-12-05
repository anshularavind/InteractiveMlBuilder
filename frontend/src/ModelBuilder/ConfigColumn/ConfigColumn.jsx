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
  blockCount,
  setBlockCount,
  createLayers,
  removeLastBlock,
  layers,
  trainInputs,
  setTrainInputs,
  loadModelConfig,
}) {
  const [blockInputs, setBlockInputs] = useState({
    outputSize: 0,
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

  // Handle JSON configuration (if any)
  const handleJsonConfig = (jsonConfig) => {
    setSelectedDataset(jsonConfig.dataset);
    setTrainInputs({
      lr: jsonConfig.lr,
      batch_size: jsonConfig.batch_size,
      epochs: jsonConfig.epochs,
    });
  
    setDatasetSizes({
      inputSize: jsonConfig.input,
      outputSize: jsonConfig.output,
    });
  
    

    let layerConfig;
  
    const processedLayers = [];
    let previousConvLayerIndex = null;

  
    jsonConfig.blocks.forEach((block, index) => {
      const inputSize = index === 0 
        ? jsonConfig.input 
        : processedLayers[processedLayers.length - 1]?.params?.output_size || jsonConfig.input;
      
        const SCALING_CONSTANT = 25;
        const log_base = Math.log(1.5);
        const log_base_pool = Math.log(3);
        const inputSizeScaled =
          (Math.log((inputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
        const outputSizeScaled =
          (Math.log((block.params.output_size + 1) || 1) * SCALING_CONSTANT) / log_base;

        console.log("outputSizeScaled", outputSizeScaled)
      
      if (block.block === "FcNN") {
        let hiddenSizeScaled;
        if (blockInputs.numHiddenLayers === 0) {
          hiddenSizeScaled = (inputSizeScaled + outputSizeScaled) / 2;
        } else {
          hiddenSizeScaled = (Math.log((blockInputs.hiddenSize + 1) || 1) * SCALING_CONSTANT) / log_base;
        }
        const numHiddenLayersScaled = blockInputs.numHiddenLayers * SCALING_CONSTANT;
        const trapHeight = SCALING_CONSTANT * 2;
        
        layerConfig = {
          id: `block-${processedLayers.length}`,
          name: `${processedLayers.length + 1}`,
          type: "FcNN",
          params: {
            input_size: inputSize,
            ...block.params
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
        
        processedLayers.push(layerConfig);
        previousConvLayerIndex = null;
        
      } else if (block.block === "Conv") {
        console.log("block", block)

        const outputSize = block.params.output_size;
        const outputSizeScaled =
          (Math.log((outputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
        const poolSize = Math.sqrt(block.params.output_size);
        const poolSizeScaled =
          (Math.log((poolSize + 1) || 1) * SCALING_CONSTANT) / log_base_pool;
        const kernelSize = block.params.kernel_size;
        const kernelSizeScaled =
          (Math.log((kernelSize + 1) || 1) * SCALING_CONSTANT) / log_base;
        const numKernels = block.params.num_kernels;

        let outputLength;

        if (jsonConfig.input === 784) {
          outputLength = Math.floor(28 - kernelSize) + 1;
        } else if (jsonConfig.input === 3072) {
          outputLength = Math.floor(32 - kernelSize) + 1;
        }

        const outputLengthScaled = 
        (Math.log((outputLength + 1) || 1) * SCALING_CONSTANT) / log_base_pool;

        console.log("outputLength", outputLength)
        
        layerConfig = {
          id: `block-${processedLayers.length}`,
          name: `${processedLayers.length + 1}`,
          type: "Conv",
          params: {
            input_size: inputSize,
            output_size: outputSize,
            kernel_size: kernelSize,
            num_kernels: numKernels,
            stride: block.params.stride || 1,
            padding: block.params.padding || Math.floor(kernelSize / 2),
          },
          visParams: {
            kernel_size: kernelSizeScaled,
            poolingBlock: { smallBlock: poolSizeScaled, largeBlock: outputLengthScaled },
            width: kernelSizeScaled + numKernels * 5 + outputLengthScaled + 10,
          },
        };
        
        processedLayers.push(layerConfig);
        previousConvLayerIndex = processedLayers.length - 1;
      }
    });
  
    createLayers(processedLayers);
    setBlockCount(processedLayers.length);
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
    const log_base_pool = Math.log(3);
    const inputSizeScaled =
      (Math.log((inputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
    const outputSizeScaled =
      (Math.log(((blockInputs.outputSize ** 2) + 1) || 1) * SCALING_CONSTANT) / log_base;

    let newLayer;

    if (selectedLayer === "FcNN") {
      let hiddenSizeScaled;
      if (blockInputs.numHiddenLayers === 0) {
        hiddenSizeScaled = (inputSizeScaled + outputSizeScaled) / 2;
      } else {
        hiddenSizeScaled = (Math.log((blockInputs.hiddenSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      }
      const numHiddenLayersScaled = blockInputs.numHiddenLayers * SCALING_CONSTANT;
      const trapHeight = SCALING_CONSTANT * 2;

      newLayer = {
        id: `block-${newBlockId}`,
        name: `${newBlockId + 1}`,
        type: "FcNN",
        params: {
          input_size: inputSize,
          output_size: blockInputs.outputSize,
          hidden_size: blockInputs.hiddenSize,
          num_hidden_layers: blockInputs.numHiddenLayers,
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
    } else if (selectedLayer === "Conv") {
      const outputSize = blockInputs.outputSize;
      const poolSize = Math.sqrt(blockInputs.outputSize);
      const outputSizeScaled =
        (Math.log((outputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      const poolSizeScaled =
        (Math.log((poolSize + 1) || 1) * SCALING_CONSTANT) / log_base_pool;
      const kernelSize = blockInputs.kernelSize;
      const kernelSizeScaled =
        (Math.log((kernelSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      const numKernels = blockInputs.numKernels;
      let outputLength;

      if (datasetSizes.inputSize === 784) {
        outputLength = Math.floor(28 - blockInputs.kernelSize) + 1;
      } else if (datasetSizes.inputSize === 3072) {
        outputLength = Math.floor(32 - blockInputs.kernelSize) + 1;
      }

      const outputLengthScaled =
        (Math.log((outputLength + 1) || 1) * SCALING_CONSTANT) / log_base_pool;

      newLayer = {
        id: `block-${newBlockId}`,
        name: `${newBlockId + 1}`,
        type: "Conv",
        params: {
          input_size: inputSize,
          output_size: blockInputs.outputSize ** 2,
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
    }

    createLayers([newLayer]);

    setBlockInputs({
      outputSize: 0,
      hiddenSize: 0,
      numHiddenLayers: 0,
      kernelSize: 3,
      numKernels: 1,
    });
    setSelectedLayer(null);
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
