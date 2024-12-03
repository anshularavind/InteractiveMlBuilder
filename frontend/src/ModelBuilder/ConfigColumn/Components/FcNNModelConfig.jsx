import "../../ModelBuilder.css";

function FcNNModelConfig({ blockInputs, handleInputChange, createLayers }) {
  return (
    <div>
      <div>
        <label>Output Size: </label>
        <br/>
        <input
          type="number"
          value={blockInputs.outputSize}
          onChange={(e) => handleInputChange("outputSize", e.target.value)}
          className="input"
        />
      </div>
      <div>
        <label>Hidden Size: </label>
        <br/>
        <input
          type="number"
          value={blockInputs.hiddenSize}
          onChange={(e) => handleInputChange("hiddenSize", e.target.value)}
          className="input"
        />
      </div>
      <div>
        <label>Num Hidden Layers: </label>
        <br/>
        <input
          type="number"
          value={blockInputs.numHiddenLayers}
          onChange={(e) => handleInputChange("numHiddenLayers", e.target.value)}
          className="input"
        />
      </div>
      <button className="addButton" onClick={createLayers}>
        Add Block
      </button>
    </div>
  );
}

export default FcNNModelConfig;
