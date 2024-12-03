import React, { useState, useEffect } from "react";
import DatasetSelection from "./Components/DatasetSelection";
import LayerSelection from "./Components/LayerSelection";
import ModelConfig from "./Components/ModelConfig";
import ConvModelConfig from "./Components/ConvModelConfig";
import Train from "./Components/Train";
import "../ModelBuilder.css";

function ConfigColumn({
  selectedDataset,
  setSelectedDataset,
  datasetDropdownOpen,
  toggleDatasetDropdown,
  datasetItems,
  handleDatasetClick,
  layerItems,
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
    kernelSize: 0,
    numKernels: 0,
  });

  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);
  const [blockCount, setBlockCount] = useState(0);
  const [datasetSizes, setDatasetSizes] = useState({ inputSize: 0, outputSize: 0 });

  // Add the handleJsonConfig method here
  const handleJsonConfig = (jsonConfig) => {
    // Set dataset and training parameters
    setSelectedDataset(jsonConfig.dataset);
    // selectedDataset = jsonConfig.dataset;
    
    setTrainInputs({
      lr: jsonConfig.lr,
      batch_size: jsonConfig.batch_size, 
      epochs: jsonConfig.epochs
    });

    // Set initial dataset sizes
    setDatasetSizes({
      inputSize: jsonConfig.input,
      outputSize: jsonConfig.output
    });

    // Process each block
    const processedLayers = jsonConfig.blocks.map((block, index) => {
      const SCALING_CONSTANT = 25;
      const log_base = Math.log(1.5);
      
      const inputSize = index === 0 ? jsonConfig.input : 
        jsonConfig.blocks[index-1].params.output_size;
      
      const inputSizeScaled = (Math.log((inputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      const outputSizeScaled = (Math.log((block.params.output_size + 1) || 1) * 
        SCALING_CONSTANT) / log_base;

      let layerConfig = {
        id: `block-${index}`,
        name: `${index + 1}`,
        type: block.block,
        params: {
          input_size: inputSize,
          ...block.params
        }
      };

      if (block.block === 'FcNN') {
        const hiddenSizeScaled = (Math.log((block.params.hidden_size + 1) || 1) * 
          SCALING_CONSTANT) / log_base;
        const numHiddenLayersScaled = block.params.num_hidden_layers * SCALING_CONSTANT;
        
        layerConfig = {
          ...layerConfig,
          leftTrapezoid: { base: inputSizeScaled, height: SCALING_CONSTANT * 2 },
          rightTrapezoid: { base: outputSizeScaled, height: SCALING_CONSTANT * 2 },
          middleRectangle: { 
            width: numHiddenLayersScaled, 
            height: hiddenSizeScaled 
          }
        };
      } else if (block.block === 'Conv') {
        const kernelSizeScaled = (Math.log((block.params.kernel_size + 1) || 1) * 
          SCALING_CONSTANT) / log_base;
          
        layerConfig = {
          ...layerConfig,
          leftTrapezoid: { base: inputSizeScaled, height: SCALING_CONSTANT },
          rightTrapezoid: { base: outputSizeScaled, height: SCALING_CONSTANT },
          middleRectangle: { 
            width: kernelSizeScaled, 
            height: kernelSizeScaled 
          }
        };
      }

      return layerConfig;
    });

    // Update layers state
    createLayers(processedLayers);
    setBlockCount(processedLayers.length);
  };

  useEffect(() => {
    if (selectedDataset === "MNIST") {
      setDatasetSizes({ inputSize: 784, outputSize: 10 });
    } else if (selectedDataset === "CIFAR 10") {
      setDatasetSizes({ inputSize: 3072, outputSize: 10 });
    } else {
      setDatasetSizes({ inputSize: 0, outputSize: 0 });
    }
  }, [selectedDataset]);

  const handleBlockInputChange = (field, value) => {
    setBlockInputs((prevInputs) => ({
      ...prevInputs,
      [field]: Math.max(0, parseInt(value) || 0),
    }));
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
    const inputSizeScaled = (Math.log((inputSize + 1) || 1) * SCALING_CONSTANT) / log_base;
    const outputSizeScaled = (Math.log((blockInputs.outputSize + 1) || 1) * SCALING_CONSTANT) / log_base;

    let newLayer;

    if (selectedLayer === 'FcNN') {
      const hiddenSizeScaled = (Math.log((blockInputs.hiddenSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      const numHiddenLayersScaled = blockInputs.numHiddenLayers * SCALING_CONSTANT;
      const trapHeight = SCALING_CONSTANT * 2;

      newLayer = {
        id: `block-${newBlockId}`,
        name: `${newBlockId + 1}`,
        type: 'FcNN',
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
    } else if (selectedLayer === 'Conv') {
      const kernelSize = blockInputs.kernelSize;
      const kernelSizeScaled = (Math.log((kernelSize + 1) || 1) * SCALING_CONSTANT) / log_base;
      const numKernels = blockInputs.numKernels;
      newLayer = {
        id: `block-${newBlockId}`,
        name: `${newBlockId + 1}`,
        type: 'Conv',
        params: {
          input_size: inputSize,
          output_size: (kernelSize * kernelSize),
          kernel_size: kernelSize,
          num_kernels: numKernels,
          stride: 0,
          padding: 0
        },
        leftTrapezoid: { base: inputSizeScaled, height: SCALING_CONSTANT },
        rightTrapezoid: { base: outputSizeScaled, height: SCALING_CONSTANT },
        middleRectangle: { width: kernelSizeScaled, height: kernelSizeScaled },
      };
    }

    createLayers([newLayer]);

    setBlockInputs({
      outputSize: 0,
      hiddenSize: 0,
      numHiddenLayers: 0,
      kernelSize: 0,
      numKernels: 0,
    });
    setSelectedLayer(null);
  };

  useEffect(() => {
    let cachedModelConfig = loadModelConfig();
    if (cachedModelConfig !== null && layers.length === 0 && cachedModelConfig.blocks.length > 0) {
      handleJsonConfig(cachedModelConfig);
    }
  }, [layers, loadModelConfig]);

  return (
    <div className="inputBlock">
      <div className="inputBlockHeader">
        <h1>
          <b>Configuration</b>
        </h1>
        <h2>

        </h2>
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
      {selectedDataset && (
        <div className="inputBlockContent">
          <h2>
            Add Blocks
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
          {selectedLayer === "Conv" && (
            <ConvModelConfig
              blockInputs={blockInputs}
              handleInputChange={handleBlockInputChange}
              createLayers={addBlock}
            ></ConvModelConfig>
          )}
          {/* Remove Last Block Button */}
          <br/>
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
          <br/>
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

