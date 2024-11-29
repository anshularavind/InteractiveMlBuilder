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
  handleInputChange,
  createLayers,
}) {
  const [modules, setModules] = useState([]);

  const addModelModule = () => {
    setModules((prevModules) => [
      ...prevModules,
      {
        id: prevModules.length,
        selectedLayer: null,
        layerDropdownOpen: false,
        blockInputs: { inputSize: 0, outputSize: 0, hiddenSize: 0, numHiddenLayers: 0 },
        isOpen: false, 
      },
    ]);
  };

  const toggleLayerDropdown = (id) => {
    setModules((prevModules) =>
      prevModules.map((module) =>
        module.id === id
          ? { ...module, layerDropdownOpen: !module.layerDropdownOpen }
          : module
      )
    );
  };

  const handleLayerClick = (id, item) => {
    setModules((prevModules) =>
      prevModules.map((module) =>
        module.id === id 
          ? { ...module, selectedLayer: item.value, layerDropdownOpen: false } 
          : module
      )
    );
  };

  const handleModuleInputChange = (id, field, value) => {
    setModules((prevModules) =>
      prevModules.map((module) =>
        module.id === id
          ? {
              ...module,
              blockInputs: {
                ...module.blockInputs,
                [field]: Math.max(0, parseInt(value) || 0),
              },
            }
          : module
      )
    );
  };

  const createModuleLayers = (id) => {
    const module = modules.find((mod) => mod.id === id);
    if (!module) return;
  
    const leftTrapezoidBase = module.blockInputs.inputSize / 10;
    const rightTrapezoidBase = module.blockInputs.outputSize / 10;
    const middleRectangleHeight = module.blockInputs.hiddenSize / 10;
    const middleRectangleWidth = module.blockInputs.numHiddenLayers * 10;
  
    const newLayer = {
      id: `${id}`,
      name: `Block ${id + 1}`,
      position: { x: 0, y: 0 },
      leftTrapezoid: { base: leftTrapezoidBase, height: middleRectangleHeight },
      rightTrapezoid: { base: rightTrapezoidBase, height: middleRectangleHeight },
      middleRectangle: { width: middleRectangleWidth, height: middleRectangleHeight },
    };
  
    createLayers([newLayer]);
  };

  const toggleModuleVisibility = (id) => {
    setModules((prevModules) =>
      prevModules.map((module) =>
        module.id === id
          ? { ...module, isOpen: !module.isOpen }
          : module
      )
    );
  };

  return (
    <div className="inputBlock">
      <div className="inputBlockHeader">
        <h1><b>Model Configuration</b></h1>
        <h2><u>Select Dataset</u></h2>
        <DatasetSelection
          selectedItem={selectedDataset}
          dropdownOpen={datasetDropdownOpen}
          toggleDropdown={toggleDatasetDropdown}
          datasetItems={datasetItems}
          handleItemClick={handleDatasetClick}
        />
        <h4>Input Size:</h4>
        <h4>Output Size: </h4>
        <h2><u>Add Blocks</u></h2>
        <button className="addButton" onClick={addModelModule}>
          +
        </button>
      </div>

      <div className="inputBlockContent">
        {modules.map((module) => (
          <div key={module.id} className="module">
            <h3
              className="moduleHeader"
              onClick={() => toggleModuleVisibility(module.id)}
            >
              Configure Block {module.id + 1}
            </h3>
            {module.isOpen && (
              <div className="moduleContent">
                <LayerSelection
                  selectedItem={module.selectedLayer}
                  dropdownOpen={module.layerDropdownOpen}
                  toggleDropdown={() => toggleLayerDropdown(module.id)}
                  layerItems={layerItems}
                  handleItemClick={(item) => handleLayerClick(module.id, item)}
                />
                <ModelConfig
                  blockInputs={module.blockInputs}
                  handleInputChange={(field, value) =>
                    handleModuleInputChange(module.id, field, value)
                  }
                  createLayers={() => createModuleLayers(module.id)}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      <Train />
    </div>
  );
}

export default ConfigColumn;