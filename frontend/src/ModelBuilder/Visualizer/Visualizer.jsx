import React, { useEffect, useState } from "react";
import "../ModelBuilder.css";

function Visualizer({ layers }) {
  const [positionedLayers, setPositionedLayers] = useState([]);

  useEffect(() => {
    let currentX = 0;
    const newPositionedLayers = layers.map((layer) => {
      const newLayer = {
        ...layer,
        position: { x: currentX, y: 0 }
      };
      currentX += layer.leftTrapezoid.height + layer.middleRectangle.width + layer.rightTrapezoid.height - (layer.leftTrapezoid.height);
      return newLayer;
    });
  
    setPositionedLayers(newPositionedLayers);
  }, [layers]);

  const renderBowtie = (bowtie) => {
    const { leftTrapezoid, rightTrapezoid, middleRectangle, name } = bowtie;
    const width = leftTrapezoid.height + middleRectangle.width + rightTrapezoid.height;
    const height = Math.max((leftTrapezoid.base), middleRectangle.height, rightTrapezoid.base) + 20;

    return (
      <svg
        width={width}
        height={height}
        viewBox={`${-leftTrapezoid.base} ${-middleRectangle.height / 2 - 10} ${width} ${height}`}
        style={{ overflow: "visible" }}
      >
        <polygon
          points={`0,${-middleRectangle.height / 2} ${-leftTrapezoid.height},${-leftTrapezoid.base / 2} ${-leftTrapezoid.height},${leftTrapezoid.base / 2} 0,${middleRectangle.height / 2}`}
          fill="blue"
        />
        <rect
          x="0"
          y={-middleRectangle.height / 2}
          width={middleRectangle.width}
          height={middleRectangle.height}
          fill="green"
        />
        <text
          x={middleRectangle.width / 2}
          y="0"
          textAnchor="middle"
          alignmentBaseline="middle"
          fill="white"
          fontSize="12px"
        >
          {name}
        </text>
        <polygon
          points={`${middleRectangle.width},${-middleRectangle.height / 2} ${middleRectangle.width + rightTrapezoid.height},${-rightTrapezoid.base / 2} ${middleRectangle.width + rightTrapezoid.height},${rightTrapezoid.base / 2} ${middleRectangle.width},${middleRectangle.height / 2}`}
          fill="red"
        />
      </svg>
    );
  };

  return (
    <div className="visual-flowchart" style={{ position: "relative", height: "100%", width: "100%" }}>
      {positionedLayers.map((layer) => (
        <div
          key={layer.id}
          style={{
            position: "absolute",
            left: "50%",
            top: "50%",
            transform: `translate(${layer.position.x}px, ${layer.position.y}px)`,
          }}
        >
          {renderBowtie(layer)}
        </div>
      ))}
    </div>
  );
}

export default Visualizer;