import React, { useState } from "react";
import Draggable from "react-draggable";

function FlowChartOrganizer() {
  const [layers, setLayers] = useState([]);
  const [blockInputs, setBlockInputs] = useState({
    inputSize: 0,
    outputSize: 0,
    hiddenSize: 0,
    numHiddenLayers: 1,
  });

  const handleInputChange = (field, value) => {
    setBlockInputs({ ...blockInputs, [field]: value });
  };

  const createLayers = () => {
    const newLayers = [];
    const layerHeight = 50; // Height of each layer
    const overlapOffsetY = 10; // Vertical overlap amount
    const overlapOffsetX = 15; // Horizontal overlap amount
    const totalHeight =
      blockInputs.numHiddenLayers * (layerHeight - overlapOffsetY);
    const areaHeight = 500; // Approximate height of Layers Area
    const startY = (areaHeight - totalHeight) / 2; // Center layers vertically
    const startX = -((blockInputs.numHiddenLayers - 1) * overlapOffsetX) / 2; // Center layers horizontally
  
    for (let i = 0; i < blockInputs.numHiddenLayers; i++) {
      newLayers.push({
        id: layers.length + i,
        name: `Layer ${i + 1}`,
        position: {
          x: startX + i * overlapOffsetX,
          y: startY + i * (layerHeight - overlapOffsetY),
        },
      });
    }
    setLayers([...layers, ...newLayers]);
  };
  
  
  
  

  const resetLayers = () => {
    setLayers([]);
    setBlockInputs({
      inputSize: 0,
      outputSize: 0,
      hiddenSize: 0,
      numHiddenLayers: 1,
    });
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
    <div style={styles.container}>
      {/* Input Section */}
      <div style={styles.inputArea}>
        <div style={styles.label}>Input</div>
        <div style={styles.trapezoid}></div>
      </div>

      {/* Arrow from Input to Layers */}
      {layers.length > 0 && (
        <div style={styles.arrowContainer}>
          <div style={styles.arrowLine}></div>
          <div style={styles.arrowHead}></div>
        </div>
      )}

      {/* Middle Layers Area */}
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

      {/* Arrow from Layers to Output */}
      {layers.length > 0 && (
        <div style={styles.arrowContainer}>
          <div style={styles.arrowLine}></div>
          <div style={styles.arrowHead}></div>
        </div>
      )}

      {/* Output Section */}
      <div style={styles.outputArea}>
        <div style={styles.label}>Output</div>
        <div style={styles.trapezoid}></div>
      </div>

      {/* Input Controls */}
      <div style={styles.inputBlock}>
        <h4>Configure Inputs</h4>
        <div>
          <label>Input Size: </label>
          <input
            type="number"
            value={blockInputs.inputSize}
            onChange={(e) =>
              handleInputChange("inputSize", parseInt(e.target.value) || 0)
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
              handleInputChange("outputSize", parseInt(e.target.value) || 0)
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
              handleInputChange("hiddenSize", parseInt(e.target.value) || 0)
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
              handleInputChange(
                "numHiddenLayers",
                Math.max(1, parseInt(e.target.value) || 1)
              )
            }
            style={styles.input}
          />
        </div>
        <button style={styles.addButton} onClick={createLayers}>
          Create Layers
        </button>
        <button style={styles.resetButton} onClick={resetLayers}>
          Reset
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "grid",
    gridTemplateColumns: "1fr 0.15fr 6fr 0.15fr 1fr 1.5fr", // Input, Arrow1, Layers, Arrow2, Output, Controls
    width: "95%",
    height: "90vh",
    margin: "auto",
    fontFamily: "'Poppins', sans-serif",
    border: "1px solid #ddd",
    borderRadius: "10px",
    overflow: "hidden",
    gap: "10px",
  },
  inputArea: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  },
  outputArea: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  },
  layersArea: {
    backgroundColor: "#f9f9f9",
    border: "1px solid #ddd",
    borderRadius: "10px",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "column",
    overflow: "hidden",
    position: "relative",
    height: "100%", // Ensure consistent height for centering
  },
  trapezoid: {
    width: "300px",
    height: "85vh",
    backgroundColor: "#90caf9",
    clipPath: "polygon(20% 0, 80% 0, 100% 100%, 0% 100%)",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  arrowContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  arrowLine: {
    width: "50px",
    height: "2px",
    backgroundColor: "black",
  },
  arrowHead: {
    width: "0",
    height: "0",
    borderLeft: "7px solid black",
    borderTop: "4px solid transparent",
    borderBottom: "4px solid transparent",
  },
  label: {
    marginBottom: "10px",
    fontWeight: "bold",
    fontSize: "1rem",
    textAlign: "center",
    color: "#333",
  },
  inputBlock: {
    backgroundColor: "#f0f8ff",
    padding: "20px",
    border: "1px solid #ddd",
    borderRadius: "5px",
    display: "flex",
    flexDirection: "column",
    alignItems: "flex-start",
    gap: "10px",
  },
  input: {
    width: "100%",
    padding: "5px",
    fontSize: "1rem",
    marginTop: "5px",
    boxSizing: "border-box",
  },
  addButton: {
    width: "100%",
    padding: "10px",
    margin: "10px 0",
    fontSize: "1rem",
    backgroundColor: "#4a90e2",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  resetButton: {
    width: "100%",
    padding: "10px",
    margin: "10px 0",
    fontSize: "1rem",
    backgroundColor: "#f44336",
    color: "#fff",
    border: "none",
    borderRadius: "5px",
    cursor: "pointer",
  },
  layer: {
    width: "120px",
    height: "50px",
    backgroundColor: "#e3f2fd",
    color: "#333",
    textAlign: "center",
    lineHeight: "50px",
    border: "1px solid #90caf9",
    borderRadius: "5px",
    position: "absolute",
  },
};

export default FlowChartOrganizer;
