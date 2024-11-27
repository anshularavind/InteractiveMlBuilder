import React, { useState } from "react";
import Draggable from "react-draggable";
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
      <div className="headerContainer">
        {/* Add buttons or other header content here */}
      </div>

      <div className="container">
        <div className="inputBlock">
          <div className="dropdown">
            <button className="dropdownButton" onClick={toggleDropdown}>
              {selectedItem ? `Selected: ${selectedItem}` : "Select Dataset"}
            </button>
            {dropdownOpen && (
              <div className="dropdownContent">
                {items.map((item) => (
                  <div
                    key={item.value}
                    className="dropdownItem"
                    onClick={() => handleItemClick(item)}
                  >
                    {item.label}
                  </div>
                ))}
              </div>
            )}
          </div>
          <h4>Configure Inputs</h4>
          <div>
            <label>Input Size: </label>
            <input
              type="number"
              value={blockInputs.inputSize}
              onChange={(e) => handleInputChange("inputSize", e.target.value)}
              className="input"
            />
          </div>
          <div>
            <label>Output Size: </label>
            <input
              type="number"
              value={blockInputs.outputSize}
              onChange={(e) => handleInputChange("outputSize", e.target.value)}
              className="input"
            />
          </div>
          <div>
            <label>Hidden Size: </label>
            <input
              type="number"
              value={blockInputs.hiddenSize}
              onChange={(e) => handleInputChange("hiddenSize", e.target.value)}
              className="input"
            />
          </div>
          <div>
            <label>Num Hidden Layers: </label>
            <input
              type="number"
              value={blockInputs.numHiddenLayers}
              onChange={(e) =>
                handleInputChange("numHiddenLayers", e.target.value)
              }
              className="input"
            />
          </div>
          <button className="addButton" onClick={createLayers}>
            Create Layers
          </button>
        </div>

        <div className="layersArea">
          {layers.map((layer) => (
            <Draggable
              key={layer.id}
              position={layer.position}
              onStop={(e, data) => onLayerDragStop(layer.id, data)}
            >
              <div className="layer">{layer.name}</div>
            </Draggable>
          ))}
        </div>
      </div>
    </div>
  );
}

export default ModelBuilder;
