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
  selectedLayer,
  layerDropdownOpen,
  toggleLayerDropdown,
  layerItems,
  handleLayerClick,
  blockInputs,
  handleInputChange,
  createLayers,
  removeLastLayer,  // Assuming you have removeLastLayer function passed in
}) {
  return (
    <div className="inputBlock">
      {/* Heading for Dataset Selection */}
      <h2>Select Dataset:</h2>
      <DatasetSelection
        selectedItem={selectedDataset}
        dropdownOpen={datasetDropdownOpen}
        toggleDropdown={toggleDatasetDropdown}
        datasetItems={datasetItems}
        handleItemClick={handleDatasetClick}
      />

      {/* Heading for Layer Selection */}
      <h2>Add Layers:</h2>
      <LayerSelection
        selectedItem={selectedLayer}
        dropdownOpen={layerDropdownOpen}
        toggleDropdown={toggleLayerDropdown}
        layerItems={layerItems}
        handleItemClick={handleLayerClick}
      />

      <ModelConfig
        blockInputs={blockInputs}
        handleInputChange={handleInputChange}
        createLayers={createLayers}
      />
      <Train />

      {/* Buttons */}
      <button className="addButton" onClick={createLayers}>
        Create Layers
      </button>
      <button className="removeButton" onClick={removeLastLayer}>
        Remove Last Layer
      </button>
    </div>
  );
}

export default ConfigColumn;