import React, { useState } from "react";
import Draggable from "react-draggable";

function ModelBuilder() {
  const [selectedItem, setSelectedItem] = useState(null); // Track the selected item
  const [dropdownOpen, setDropdownOpen] = useState(false); // Track dropdown state

  const items = [
    { value: "MNIST", label: "MNIST" },
    { value: "CIFAR 10", label: "CIFAR 10" },
  ];

  const handleItemClick = (item) => {
    setSelectedItem(item.value); // Set the selected item
    setDropdownOpen(false); // Close the dropdown after selection
  };

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen); // Toggle the dropdown open/close
  };

  const [layers, setLayers] = useState([]);
  const [blockInputs, setBlockInputs] = useState({
    inputSize: 0,
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 0,
  });

  const handleInputChange = (field, value) => {
    // Prevent negative values
    const sanitizedValue = Math.max(0, parseInt(value) || 0);
    setBlockInputs({ ...blockInputs, [field]: sanitizedValue });
  };

  const createLayers = () => {
    const newLayers = [];
    for (let i = 0; i < blockInputs.numHiddenLayers; i++) {
      newLayers.push({
        id: layers.length + i,
        name: `Layer ${i + 1}`,
        position: { x: 0, y: i * 60 }, // Staggered vertically
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
      {/* Header with Back to Home and Profile Buttons */}
      <div style={styles.headerContainer}>
       
        
      </div>

      <div style={styles.container}>
        {/* Input Block */}
        <div style={styles.inputBlock}>
          <div className="dropdown" style={styles.dropdown}>
            <button
              className="dropdown-button"
              style={styles.dropdownButton}
              onClick={toggleDropdown}
            >
              {selectedItem ? `Selected: ${selectedItem}` : "Select Dataset"}
            </button>
            {dropdownOpen && (
              <div className="dropdown-content" style={styles.dropdownContent}>
                {items.map((item) => (
                  <div
                    key={item.value}
                    style={styles.dropdownItem}
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
              onChange={(e) =>
                handleInputChange("inputSize", e.target.value)
              }
              style={styles.input}
            />
          </div>
          <div>
            <label>Output Size: </label>
            <input
              type="number"
              value={blockInputs.outputSize}
              onChange={(e) =>
                handleInputChange("outputSize", e.target.value)
              }
              style={styles.input}
            />
          </div>
          <div>
            <label>Hidden Size: </label>
            <input
              type="number"
              value={blockInputs.hiddenSize}
              onChange={(e) =>
                handleInputChange("hiddenSize", e.target.value)
              }
              style={styles.input}
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
              style={styles.input}
            />
          </div>
          <button style={styles.addButton} onClick={createLayers}>
            Create Layers
          </button>
        </div>

        {/* Draggable Layers */}
        <div style={styles.layersArea}>
          {layers.map((layer) => (
            <Draggable
              key={layer.id}
              position={layer.position}
              onStop={(e, data) => onLayerDragStop(layer.id, data)}
            >
              <div style={styles.layer}>{layer.name}</div>
            </Draggable>
          ))}
        </div>
      </div>
    </div>
  );
}

const styles = {
  
  headerButton: {
    padding: "5px 15px",
    fontSize: "1rem",
    color: "#fff",
    backgroundColor: "#357abd",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  container: {
    display: "flex",
    height: "100vh",
    fontFamily: "'Poppins', sans-serif",
  },
  inputBlock: {
    width: "25%",
    padding: "20px",
    backgroundColor: "#f0f8ff",
    borderRight: "2px solid #ddd",
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: "10px",
  },
  layersArea: {
    width: "75%",
    position: "relative",
    backgroundColor: "#f9f9f9",
    overflow: "hidden",
    padding: "10px",
  },
  input: {
    width: "100%",
    padding: "5px",
    fontSize: "1rem",
    marginTop: "5px",
    boxSizing: "border-box",
  },
  addButton: {
    padding: "10px 20px",
    fontSize: "1rem",
    color: "#fff",
    backgroundColor: "#4a90e2",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    marginTop: "10px",
  },
  layer: {
    width: "150px",
    padding: "10px",
    textAlign: "center",
    backgroundColor: "#e3f2fd",
    border: "1px solid #90caf9",
    borderRadius: "5px",
    cursor: "grab",
    boxShadow: "0 2px 5px rgba(0,0,0,0.2)",
    position: "absolute",
  },
  dropdown: {
    marginBottom: "10px",
    position: "relative",
    width: "100%",
  },
  dropdownButton: {
    width: "100%",
    padding: "10px",
    backgroundColor: "#357abd",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
    textAlign: "center",
  },
  dropdownContent: {
    position: "absolute",
    backgroundColor: "#fff",
    border: "1px solid #ddd",
    borderRadius: "5px",
    marginTop: "5px",
    zIndex: 100,
    padding: "10px",
    width: "100%",
    cursor: "pointer",
  },
  dropdownItem: {
    padding: "5px 10px",
    fontSize: "1rem",
    color: "#333",
    cursor: "pointer",
    borderBottom: "1px solid #ddd",
  },
};

export default ModelBuilder;