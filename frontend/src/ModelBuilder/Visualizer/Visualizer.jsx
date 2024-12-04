import React, { useEffect, useState, useRef } from "react";
import "../ModelBuilder.css";
import FcNNBlock from "./Components/FcNNBlock";
import ConvBlock from "./Components/ConvBlock";

function Visualizer({ layers }) {
  const { visParams, name, params } = layers;
  const [positionedLayers, setPositionedLayers] = useState([]);
  const containerRef = useRef(null);
  const [containerDimensions, setContainerDimensions] = useState({
    width: 0,
    height: 0,
  });

  const [hoveredBlock, setHoveredBlock] = useState(null);

  const handleMouseEnter = (blockId) => {
    setHoveredBlock(blockId);
  };

  const handleMouseLeave = () => {
    setHoveredBlock(null);
  };


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
    } else if (layers.length > 0 && containerDimensions.width > 0) {
      const spacing = 50;

      // Calculate total unscaled width of all layers
      const totalUnscaledWidth = layers.reduce(
        (acc, layer) => acc + layer.visParams.width + spacing,
        0
      );

      // Calculate scaling factor to fit within the container
      const scalingFactor = Math.min(
        1,
        (containerDimensions.width - spacing * layers.length) / totalUnscaledWidth
      );

      // Scale and position layers
      let currentX = spacing; // Start with spacing to ensure padding at the start
      const scaledLayers = layers.map((layer) => {
        // Scale visParams
        const scaledVisParams = {};
        for (let key in layer.visParams) {
          if (layer.visParams.hasOwnProperty(key)) {
            const value = layer.visParams[key];
            if (typeof value === "object") {
              scaledVisParams[key] = {};
              for (let prop in value) {
                if (
                  value.hasOwnProperty(prop) &&
                  typeof value[prop] === "number"
                ) {
                  scaledVisParams[key][prop] = value[prop] * scalingFactor;
                } else {
                  scaledVisParams[key][prop] = value[prop];
                }
              }
            } else if (typeof value === "number") {
              scaledVisParams[key] = value * scalingFactor;
            } else {
              scaledVisParams[key] = value;
            }
          }
        }

        const layerWidth = scaledVisParams.width + spacing;

        const layerPosition = {
          x: currentX,
          y:
            -(scaledVisParams.middleRectangle?.height || 0) / 2 +
            containerDimensions.height / 2,
        };

        currentX += layerWidth;

        return {
          ...layer,
          position: layerPosition,
          visParams: scaledVisParams,
        };
      });

      // Calculate total scaled width of the flowchart
      const totalScaledWidth =
        scaledLayers.reduce((acc, layer) => acc + layer.visParams.width + spacing, 0) - spacing;

      // Adjust offset to ensure the flowchart is centered or aligned within bounds
      const offsetX = Math.max(0, (containerDimensions.width - totalScaledWidth) / 2);

      // Adjust positions to center the flowchart or fit it within bounds
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



  const renderBlock = (block) => {
    const { type } = block;

    if (type === "Conv") {
      return <ConvBlock layer={block} />;
    }

    return <FcNNBlock layer={block} />;
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
          <div
            className="block-container"
            style={{ position: "relative", display: "inline-block" }}
            onMouseEnter={(e) => {
              const tooltip = document.getElementById(`tooltip-${layer.id}`);
              tooltip.style.display = "block";
            }}
            onMouseLeave={(e) => {
              const tooltip = document.getElementById(`tooltip-${layer.id}`);
              tooltip.style.display = "none";
            }}
          >
            {/* Tooltip */}
            <div
              id={`tooltip-${layer.id}`}
              className="tooltip"
              
            >
              {Object.entries(layer.params).map(([key, value]) => (
                <div key={key}>
                  {key}: {value}
                </div>
              ))}
            </div>
  
            {/* Render the block */}
            {renderBlock(layer)}
          </div>
        </div>
      ))}
    </div>
  );
  
  
}

export default Visualizer;