import Draggable from "react-draggable";
import "../ModelBuilder.css";

function Visualizer({ layers, onLayerDragStop }) {
  return (
    <div className="visual-flowchart">
      {layers.map((layer) => (
        <Draggable
          key={layer.id}
          position={layer.position}
          onStop={(e, data) => onLayerDragStop(layer.id, data)}
        >
          <div className="visual-layer">
            <h3>{layer.name}</h3>
          </div>
        </Draggable>
      ))}
    </div>
  );
}

export default Visualizer;