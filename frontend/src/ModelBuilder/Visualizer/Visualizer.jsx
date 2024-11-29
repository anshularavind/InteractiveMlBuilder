import Draggable from "react-draggable";
import "../ModelBuilder.css";

function Visualizer({ layers, onLayerDragStop }) {
  const renderBowtie = (bowtie) => {
    const { leftTrapezoid, rightTrapezoid, middleRectangle, name } = bowtie;

    return (
      <svg
        width={leftTrapezoid.base+ middleRectangle.width + rightTrapezoid.base}
        height={Math.max(leftTrapezoid.height, middleRectangle.height, rightTrapezoid.height) + 20}
        viewBox={`${-leftTrapezoid.base} ${-middleRectangle.height / 2 - 10} ${leftTrapezoid.base + middleRectangle.width + rightTrapezoid.base} ${middleRectangle.height + 20}`}
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
    <div className="visual-flowchart">
      {layers.map((layer) => (
        <Draggable
          key={layer.id}
          position={layer.position}
          onStop={(e, data) => onLayerDragStop(layer.id, data)}
        >
          <div style={{ position: "absolute", background: "none" }}>
            {renderBowtie(layer)}
          </div>
        </Draggable>
      ))}
    </div>
  );
}

export default Visualizer;