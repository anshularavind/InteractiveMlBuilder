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

  const poolX = -width + params.num_kernels * 5 + convWidth + poolOffset;
  const poolY = -height + params.num_kernels * 5;

  const smallPoolX = poolX + poolingBlock.largeBlock / 2 - poolingBlock.smallBlock / 2;
  const smallPoolY = poolY + poolingBlock.largeBlock - poolingBlock.smallBlock / 2

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
            x={poolX}
            y={poolY}
            width={poolingBlock.largeBlock}
            height={poolingBlock.largeBlock}
            fill="darkorchid"
            stroke="black"
            strokeWidth={2}
          />
          <rect
            x={smallPoolX}
            y={smallPoolY}
            width={poolingBlock.smallBlock}
            height={poolingBlock.smallBlock}
            fill="lavender"
            stroke="black"
            strokeWidth={2}
          />
        </>
      )}
      <line
        x1={poolX + 2}
        y1={poolY + 2}
        x2={smallPoolX}
        y2={smallPoolY}
        stroke="white"
        strokeWidth={3}
        strokeDasharray="4,5" // Defines the dashed pattern
      />
      <line
        x1={poolX + poolingBlock.largeBlock - 2}
        y1={poolY + 2}
        x2={smallPoolX + poolingBlock.smallBlock}
        y2={smallPoolY}
        stroke="white"
        strokeWidth={3}
        strokeDasharray="4,5"
      />
      <line
        x1={poolX}
        y1={poolY + poolingBlock.largeBlock}
        x2={smallPoolX}
        y2={smallPoolY + poolingBlock.smallBlock}
        stroke="black"
        strokeWidth={3}
        strokeDasharray="4,5" // Defines the dashed pattern
      />
      <line
        x1={poolX + poolingBlock.largeBlock}
        y1={poolY + poolingBlock.largeBlock}
        x2={smallPoolX + poolingBlock.smallBlock}
        y2={smallPoolY + poolingBlock.smallBlock}
        stroke="black"
        strokeWidth={3}
        strokeDasharray="4,5"
      />
    </svg>
  );
}

export default ConvBlock;