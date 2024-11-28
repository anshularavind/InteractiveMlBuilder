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
        blockInputs: { hiddenSize: 0, numHiddenLayers: 0 },
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
        module.id === id ? { ...module, selectedLayer: item.value } : module
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

    const newLayers = [];
    for (let i = 0; i < module.blockInputs.numHiddenLayers; i++) {
      newLayers.push({
        id: `${id}-${i}`,
        name: `Layer ${id} Layer ${i + 1}`,
      });
    }
    createLayers(newLayers);
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

        <h2><u>Add Layers</u></h2>
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
              Layer {module.id + 1}
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