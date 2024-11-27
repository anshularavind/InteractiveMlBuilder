import React, { useState } from "react";
import "./ModelBuilder.css";

function FlowChartOrganizer() {
  const [layers, setLayers] = useState([]);
  const [blockInputs, setBlockInputs] = useState({
    hiddenSize: 0,
    numHiddenLayers: 1,
  });
  const [isLayersCreated, setIsLayersCreated] = useState(false); // Tracks if layers are created

  const handleInputChange = (field, value) => {
    setBlockInputs({ ...blockInputs, [field]: value });
  };

  const createLayers = () => {
    const newLayers = [];
    const layerHeight = 50;
    const layerWidth = 120;
    const overlapOffsetY = 10;
    const overlapOffsetX = 15;

    const totalHeight =
      blockInputs.numHiddenLayers * (layerHeight - overlapOffsetY);
    const totalWidth =
      blockInputs.numHiddenLayers * (layerWidth - overlapOffsetX);

    const areaHeight = 500;
    const areaWidth = 600;

    const startY = Math.max((areaHeight - totalHeight) / 2, 0);
    const startX = Math.max((areaWidth - totalWidth) / 2, 0);

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
    setIsLayersCreated(true); // Mark layers as created
  };

  const resetLayers = () => {
    setLayers([]);
    setBlockInputs({
      hiddenSize: 0,
      numHiddenLayers: 1,
    });
    setIsLayersCreated(false); // Reset the state
  };

  return (
    <div className="container">
      {/* Input Section */}
      <div className="inputArea">
        <div
          className="trapezoid"
          style={{
            width: isLayersCreated ? "200px" : "300px",
            height: isLayersCreated ? "30vh" : "40vh",
          }}
        ></div>
        <div className="trapezoidLabel">Input Layer</div>
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
        <div className="hiddenLayersLabel">Hidden Layers</div>
        {layers.map((layer) => (
          <div
            key={layer.id}
            className="layer"
            style={{
              top: `${layer.position.y}px`,
              left: `${layer.position.x}px`,
              position: "absolute",
            }}
          >
            {layer.name}
          </div>
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
        <div
          className="trapezoid"
          style={{
            width: isLayersCreated ? "200px" : "300px",
            height: isLayersCreated ? "30vh" : "40vh",
          }}
        ></div>
        <div className="trapezoidLabel">Output Layer</div>
      </div>

      {/* Predictions Box */}
      <div className="predictionsBox">
        <div className="predictionsLabel">Predictions</div>
      </div>

      {/* Input Controls */}
      <div className="inputBlock">
        <h4>Configure Inputs</h4>
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
