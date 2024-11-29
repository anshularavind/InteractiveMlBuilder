import "../../ModelBuilder.css";

function ModelConfig({ blockInputs, handleInputChange, createLayers }) {
  return (
    <div>
      <h4>Configure Block Parameters</h4>
      {["inputSize", "outputSize", "hiddenSize", "numHiddenLayers"].map((field) => (
        <div key={field}>
          <label>{field.replace(/([A-Z])/g, " $1")}: </label>
          <input
            type="number"
            value={blockInputs[field]}
            onChange={(e) => handleInputChange(field, e.target.value)}
            className="input"
          />
        </div>
      ))}
      <button className="addButton" onClick={createLayers}>
        Add Block
      </button>
    </div>
  );
}

export default ModelConfig;