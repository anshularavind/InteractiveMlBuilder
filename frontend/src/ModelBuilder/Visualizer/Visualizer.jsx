import React, { useEffect, useState, useRef } from "react";
import "../ModelBuilder.css";

function Visualizer({ layers }) {
  const [positionedLayers, setPositionedLayers] = useState([]);
  const containerRef = useRef(null);
  const [containerDimensions, setContainerDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;
        setContainerDimensions({ width, height });
      }
    };

    // Call updateDimensions initially
    updateDimensions();

    // Observe resizing
    const resizeObserver = new ResizeObserver(() => {
      updateDimensions();
    });

    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    // Cleanup observer on unmount
    return () => {
      if (containerRef.current) {
        resizeObserver.unobserve(containerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (layers.length === 0) {
        setPositionedLayers([]);
    }
    else if (layers.length > 0 && containerDimensions.width > 0) {
      // Calculate total unscaled width of all layers
      const totalUnscaledWidth = layers.reduce(
        (acc, layer) =>
          acc +
          layer.leftTrapezoid.base +
          layer.middleRectangle.width +
          layer.rightTrapezoid.base,
        0
      );

      // Calculate scaling factor to fit within the container
      const scalingFactor = Math.min(1, containerDimensions.width / totalUnscaledWidth);
      const spacing = 50;

      // Scale and position layers
      let currentX = 0;
      const scaledLayers = layers.map((layer) => {
        const scaledLeftTrapezoid = {
          base: layer.leftTrapezoid.base * scalingFactor,
          height: layer.leftTrapezoid.height * scalingFactor,
        };
        const scaledRightTrapezoid = {
          base: layer.rightTrapezoid.base * scalingFactor,
          height: layer.rightTrapezoid.height * scalingFactor,
        };
        const scaledMiddleRectangle = {
          width: layer.middleRectangle.width * scalingFactor,
          height: layer.middleRectangle.height * scalingFactor,
        };

        const layerWidth =
          scaledLeftTrapezoid.height +
          scaledMiddleRectangle.width +
          scaledRightTrapezoid.height +
          spacing;

        const layerPosition = { x: currentX, y: -scaledMiddleRectangle.height / 2 + containerDimensions.height / 2};

        currentX += layerWidth;
        return {
          ...layer,
          position: layerPosition,
          leftTrapezoid: scaledLeftTrapezoid,
          rightTrapezoid: scaledRightTrapezoid,
          middleRectangle: scaledMiddleRectangle,
        };
      });

      // After mapping through all layers, currentX now holds the total flowchart width
      const totalFlowchartWidth = currentX;

      // Calculate offset to center the flowchart
      const offsetX = (containerDimensions.width - totalFlowchartWidth) / 2;

      // Adjust positions to center the flowchart
      const positionedLayers = scaledLayers.map((layer) => ({
        ...layer,
        position: {
          x: layer.position.x + offsetX,
          y: layer.position.y,
        },
      }));

      setPositionedLayers(positionedLayers);
    }
  }, [layers, containerDimensions]);

  // Use your existing renderBowtie function without changes
  const renderBowtie = (bowtie) => {
    const { leftTrapezoid, rightTrapezoid, middleRectangle, name } = bowtie;

    const width =
      leftTrapezoid.height + middleRectangle.width + rightTrapezoid.height;
    const height = Math.max(
      leftTrapezoid.height,
      middleRectangle.height,
      rightTrapezoid.height
    );

    return (
      <svg
        width={width}
        height={height}
        viewBox={`${-leftTrapezoid.height} ${-middleRectangle.height / 2 - 10} ${width} ${height}`}
        style={{ overflow: "visible" }}
      >
        <polygon
          points={`0,${-middleRectangle.height / 2} ${-leftTrapezoid.height},${
            -leftTrapezoid.base / 2
          } ${-leftTrapezoid.height},${
            leftTrapezoid.base / 2
          } 0,${middleRectangle.height / 2}`}
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
          points={`${middleRectangle.width},${-middleRectangle.height / 2} ${
            middleRectangle.width + rightTrapezoid.height
          },${-rightTrapezoid.base / 2} ${
            middleRectangle.width + rightTrapezoid.height
          },${rightTrapezoid.base / 2} ${
            middleRectangle.width
          },${middleRectangle.height / 2}`}
          fill="red"
        />
      </svg>
    );
  };

  return (
    <div
      ref={containerRef}
      className="visual-flowchart"
      style={{ position: "relative", height: "100%", width: "100%" }}
    >
      {positionedLayers.map((layer) => (
        <div
          key={layer.id}
          style={{
            position: "absolute",
            left: `${layer.position.x}px`,
            top: `${layer.position.y}px`,
          }}
        >
          {renderBowtie(layer)}
        </div>
      ))}
    </div>
  );
}

export default Visualizer;
