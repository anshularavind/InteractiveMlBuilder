function ModelConfig({ blockInputs, handleInputChange, createLayers }) {
  return (
    <div>
      <h4>Configure Inputs</h4>
      {["inputSize", "outputSize", "hiddenSize", "numHiddenLayers"].map((field) => (
        <div key={field}>
          <label>{field.replace(/([A-Z])/g, " $1")}: </label>
          <input
            type="number"
            value={blockInputs[field]}
            onChange={(e) => handleInputChange(field, e.target.value)}
            style={styles.input}
          />
        </div>
      ))}
      <button style={styles.addButton} onClick={createLayers}>
        Create Layers
      </button>
    </div>
  );
}
