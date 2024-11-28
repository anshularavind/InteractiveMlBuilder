import React from "react";
import "../ModelBuilder.css";

function Visualizer({ layers }) {
  // Dimensions and spacing for the visual layout
  const inputLayerX = 100; // X-coordinate for the input layer
  const outputLayerX = 500; // X-coordinate for the output layer
  const trapezoidHeight = 80; // Height of the trapezoid
  const trapezoidWidthTop = 40; // Width of the top of the trapezoid
  const trapezoidWidthBottom = 150; // Width of the bottom of the trapezoid
  const layerWidth = 100; // Width of each hidden layer
  const layerHeight = 50; // Height of each hidden layer
  const layerSpacing = 30; // Vertical spacing between hidden layers
  const canvasHeight = Math.max(400, 200 + layers.length * (layerHeight + layerSpacing)); // Adjust canvas height dynamically
  const centerY = canvasHeight / 2; // Center of the canvas

  // Generate coordinates for the hidden layers
  const hiddenLayers = layers.map((layer, index) => ({
    x: (inputLayerX + outputLayerX) / 2 - layerWidth / 2, // Center hidden layers horizontally
    y: centerY - (layers.length * (layerHeight + layerSpacing)) / 2 + index * (layerHeight + layerSpacing), // Position layers vertically
    id: layer.id, // Layer ID
  }));

  return (
    <div className="visual-flowchart">
      <svg className="visual-container" width="700" height={canvasHeight}>
        {/* Input Layer (Horizontal Trapezoid) */}
        <polygon
          className="input-layer"
          points={`
            ${inputLayerX},${centerY - trapezoidWidthBottom} 
      ${inputLayerX + trapezoidHeight},${centerY - trapezoidWidthBottom} 
      ${inputLayerX + trapezoidWidthBottom},${centerY + trapezoidWidthBottom / 2} 
      ${inputLayerX},${centerY + trapezoidHeight / 2}
        `}
        />
        <text className="layer-label" x={inputLayerX + trapezoidWidthBottom / 2} y={centerY + trapezoidHeight + 10}>
          Input Layer
        </text>

        {/* Hidden Layers (Rectangles) */}
        {hiddenLayers.map((layer) => (
          <g key={layer.id}>
            <rect
              className="hidden-layer"
              x={layer.x+40}
              y={layer.y}
              width={layerWidth}
              height={layerHeight}
            />
            <text
              className="hidden-layer-label"
              x={layer.x + layerWidth / 4}
              y={layer.y + layerHeight / 2 + 5}
            >
              {layer.name}
            </text>
          </g>
        ))}

        {/* Output Layer (Horizontal Trapezoid) */}
        <polygon
          className="output-layer"
          points={`${outputLayerX},${centerY - trapezoidHeight / 2} 
                  ${outputLayerX + trapezoidWidthBottom},${centerY - trapezoidHeight / 2} 
                  ${outputLayerX + trapezoidWidthTop},${centerY + trapezoidHeight / 2} 
                  ${outputLayerX},${centerY + trapezoidHeight / 2}`}
        />
        <text className="layer-label" x={outputLayerX + trapezoidWidthBottom / 2} y={centerY + trapezoidHeight + 10}>
          Output Layer
        </text>

        {/* Straight Arrow Connections */}
        <line
          className="connection-line"
          x1={inputLayerX + trapezoidWidthTop}
          y1={centerY}
          x2={inputLayerX + 150}
          y2={centerY}
          markerEnd="url(#arrow)"
        />
        <line
          className="connection-line"
          x1={inputLayerX + 310}
          y1={centerY}
          x2={outputLayerX }
          y2={centerY}
          markerEnd="url(#arrow)"
        />

        {/* Arrowhead Definition */}
        <defs>
          <marker
            id="arrow"
            markerWidth="10"
            markerHeight="10"
            refX="6"
            refY="3"
            orient="auto"
          >
            <path d="M0,0 L0,6 L9,3 z" fill="#333" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}

export default Visualizer;
