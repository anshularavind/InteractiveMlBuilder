function Visualizer({ layers, onLayerDragStop }) {
  return (
    <div style={styles.layersArea}>
      {layers.map((layer) => (
        <Draggable
          key={layer.id}
          position={layer.position}
          onStop={(e, data) => onLayerDragStop(layer.id, data)}
        >
          <div style={styles.layer}>{layer.name}</div>
        </Draggable>
      ))}
    </div>
  );
}