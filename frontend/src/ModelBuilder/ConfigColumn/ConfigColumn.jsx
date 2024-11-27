import React, { useState } from "react";
import DatasetSelection from "./Components/DatasetSelection";
import ModelConfig from "./Components/ModelConfig";
import Train from "./Components/Train";

function ConfigColumn({
  selectedItem,
  dropdownOpen,
  toggleDropdown,
  items,
  handleItemClick,
  blockInputs,
  handleInputChange,
  createLayers,
}) {
  return (
    <div style={styles.inputBlock}>
      <DatasetSelection
        selectedItem={selectedItem}
        dropdownOpen={dropdownOpen}
        toggleDropdown={toggleDropdown}
        items={items}
        handleItemClick={handleItemClick}
      />
      <ModelConfig
        blockInputs={blockInputs}
        handleInputChange={handleInputChange}
        createLayers={createLayers}
      />
      <Train />
    </div>
  );
}