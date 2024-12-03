import React from "react";

function ConvBlock({ layer }) {
  const { visParams, name, params } = layer;
  const { kernel_size, poolingBlock, width } = visParams;

  const poolOffset = 10;

  // Define convWidth and convHeight
  const convWidth = kernel_size;
  const convHeight = kernel_size;

  const height = convHeight + params.num_kernels * 5;

  const convElements = [];
  for (let i = 0; i < params.num_kernels; i++) {
    convElements.push(
      <rect
        key={i}
        x={-width + i * 5}
        y={-height + i * 5}
        width={convWidth}
        height={convHeight}
        fill="purple"
        stroke="black"
        strokeWidth={2}
      />
    );
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`${-width} ${-height} ${width} ${height}`}
      style={{ overflow: "visible" }}
    >
      {convElements}
      <text
        x={-width/1.5}
        y={-height/2}
        textAnchor="middle"
        alignmentBaseline="middle"
        fill="white"
        fontSize="20px"
        fontWeight="bold"
      >
        {name}
      </text>
      {poolingBlock && (
        <>
          <rect
            x={-width + params.num_kernels * 5 + convWidth + poolOffset}
            y={-height + params.num_kernels * 5}
            width={poolingBlock.largeBlock}
            height={poolingBlock.largeBlock}
            fill="darkorchid"
            stroke="black"
            strokeWidth={2}
          />
          <rect
            x={
              -width +
              params.num_kernels * 5 +
              convWidth +
              poolOffset +
              poolingBlock.largeBlock / 2 -
              poolingBlock.smallBlock / 2
            }
            y={
              -height +
              params.num_kernels * 5 +
              poolingBlock.largeBlock -
              poolingBlock.smallBlock / 2
            }
            width={poolingBlock.smallBlock}
            height={poolingBlock.smallBlock}
            fill="lavender"
            stroke="black"
            strokeWidth={2}
          />
        </>
      )}
    </svg>
  );
}

export default ConvBlock;