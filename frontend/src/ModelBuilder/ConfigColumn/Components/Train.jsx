import "../../ModelBuilder.css";

function Train({ trainInputs, handleInputChange }) {
    return (
        <div className="trainParams">
            <h2 className="trainParamsHeader">Training Params</h2>
            <div className="trainInputGroup">
                <label>Learning Rate:</label>
                <input
                    type="number"
                    value={trainInputs.lr ?? 0.001}
                    onChange={(e) => handleInputChange("lr", e.target.value)}
                    className="trainInput"
                />
            </div>
            <div className="trainInputGroup">
                <label>Batch Size:</label>
                <input
                    type="number"
                    value={trainInputs.batch_size ?? 64}
                    onChange={(e) => handleInputChange("batch_size", e.target.value)}
                    className="trainInput"
                />
            </div>
            <div className="trainInputGroup">
                <label>Epochs:</label>
                <input
                    type="number"
                    value={trainInputs.epochs ?? 10}
                    onChange={(e) => handleInputChange("epochs", e.target.value)}
                    className="trainInput"
                />
            </div>
        </div>
    );
}

export default Train;
