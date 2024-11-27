import React, { useState } from "react";
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css"; // Import the external CSS file

function ModelBuilder() {
  const [selectedItem, setSelectedItem] = useState(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const items = [
    { value: "MNIST", label: "MNIST" },
    { value: "CIFAR 10", label: "CIFAR 10" },
  ];

  const handleItemClick = (item) => {
    setSelectedItem(item.value);
    setDropdownOpen(false);
  };

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };

  const [layers, setLayers] = useState([]);
  const [blockInputs, setBlockInputs] = useState({
    inputSize: 0,
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 0,
  });

  const handleInputChange = (field, value) => {
    const sanitizedValue = Math.max(0, parseInt(value) || 0);
    setBlockInputs({ ...blockInputs, [field]: sanitizedValue });
  };

  const createLayers = () => {
    const newLayers = [];
    for (let i = 0; i < blockInputs.numHiddenLayers; i++) {
      newLayers.push({
        id: layers.length + i,
        name: `Layer ${i + 1}`,
        position: { x: 0, y: i * 60 },
      });
    }
    setLayers([...layers, ...newLayers]);
  };

  const onLayerDragStop = (id, data) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === id
          ? { ...layer, position: { x: data.x, y: data.y } }
          : layer
      )
    );
  };

  return (
    <div>
      <div className="container">
        <ConfigColumn
          selectedItem={selectedItem}
          dropdownOpen={dropdownOpen}
          toggleDropdown={toggleDropdown}
          items={items}
          handleItemClick={handleItemClick}
          blockInputs={blockInputs}
          handleInputChange={handleInputChange}
          createLayers={createLayers}
        />
        <Visualizer layers={layers} onLayerDragStop={onLayerDragStop} />
      </div>
    </div>
  );
}

export default ModelBuilder;
