import "../../ModelBuilder.css";

function FcNNModelConfig({ blockInputs, handleInputChange, createLayers }) {
  return (
    <div className="fcnnConfig">
      <div className="fcnnInputGroup">
        <label htmlFor="outputSize">Output Size:</label>
        <input
          id="outputSize"
          type="number"
          value={blockInputs.outputSize}
          onChange={(e) => handleInputChange("outputSize", e.target.value)}
          className="fcnnInput"
          placeholder="Enter output size"
        />
      </div>
      <div className="fcnnInputGroup">
        <label htmlFor="hiddenSize">Hidden Size:</label>
        <input
          id="hiddenSize"
          type="number"
          value={blockInputs.hiddenSize}
          onChange={(e) => handleInputChange("hiddenSize", e.target.value)}
          className="fcnnInput"
          placeholder="Enter hidden size"
        />
      </div>
      <div className="fcnnInputGroup">
        <label htmlFor="numHiddenLayers">Num Hidden Layers:</label>
        <input
          id="numHiddenLayers"
          type="number"
          value={blockInputs.numHiddenLayers}
          onChange={(e) => handleInputChange("numHiddenLayers", e.target.value)}
          className="fcnnInput"
          placeholder="Enter number of layers"
        />
      </div>
      <button className="fcnnAddButton" onClick={createLayers}>
        Add Block
      </button>
    </div>
  );
}

export default FcNNModelConfig;
