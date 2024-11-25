import React, { useState } from "react";
import Draggable from "react-draggable";
import './ModelBuilder.css'; // Import the CSS file

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
    const layerHeight = 50;
    const overlapOffsetY = 10;
    const overlapOffsetX = 15;
    const totalHeight =
      blockInputs.numHiddenLayers * (layerHeight - overlapOffsetY);
    const areaHeight = 500;
    const startY = (areaHeight - totalHeight) / 2;
    const startX = -((blockInputs.numHiddenLayers - 1) * overlapOffsetX) / 2;

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
    <div className="container">
      {/* Input Section */}
      <div className="inputArea">
        <div className="label">Input</div>
        <div className="trapezoid"></div>
      </div>

      {/* Arrow from Input to Layers */}
      {layers.length > 0 && (
        <div className="arrowContainer">
          <div className="arrowLine"></div>
          <div className="arrowHead"></div>
        </div>
      )}

      {/* Layers Area */}
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

      {/* Arrow from Layers to Output */}
      {layers.length > 0 && (
        <div className="arrowContainer">
          <div className="arrowLine"></div>
          <div className="arrowHead"></div>
        </div>
      )}

      {/* Output Section */}
      <div className="outputArea">
        <div className="label">Output</div>
        <div className="trapezoid"></div>
      </div>

      {/* Input Controls */}
      <div className="inputBlock">
        <h4>Configure Inputs</h4>
        <div>
          <label>Input Size: </label>
          <input
            type="number"
            value={blockInputs.inputSize}
            onChange={(e) =>
              handleInputChange("inputSize", parseInt(e.target.value) || 0)
            }
            className="input"
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
            className="input"
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
            className="input"
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
            className="input"
          />
        </div>
        <button className="addButton" onClick={createLayers}>
          Create Layers
        </button>
        <button className="resetButton" onClick={resetLayers}>
          Reset
        </button>
      </div>
    </div>
  );
}

export default FlowChartOrganizer;
