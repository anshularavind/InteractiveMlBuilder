import React from "react";

function FcNNBlock({ layer }) {
  const { visParams, name } = layer;
  console.log(layer)
  const { leftTrapezoid, rightTrapezoid, middleRectangle, width } = visParams;

  const height = Math.max(
    leftTrapezoid.height,
    middleRectangle.height,
    rightTrapezoid.height
  );

  return (
    <svg
      width={width}
      height={height}
      viewBox={`${-leftTrapezoid.height} ${
        -middleRectangle.height / 2 - 10
      } ${width} ${height}`}
      style={{ overflow: "visible" }}
    >
      <polygon
        points={`0,${-middleRectangle.height / 2} ${-leftTrapezoid.height},${
          -leftTrapezoid.base / 2
        } ${-leftTrapezoid.height},${
          leftTrapezoid.base / 2
        } 0,${middleRectangle.height / 2}`}
        fill="blue"
        stroke="black"
        strokeWidth={2}
      />
      <rect
        x="0"
        y={-middleRectangle.height / 2}
        width={middleRectangle.width}
        height={middleRectangle.height}
        fill="green"
        stroke="black"
        strokeWidth={2}
      />
      <text
        x={middleRectangle.width / 2}
        y="0"
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize="20px"
        fontWeight={"bold"}
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
        stroke="black"
        strokeWidth={2}
      />
    </svg>
  );
}

export default FcNNBlock;