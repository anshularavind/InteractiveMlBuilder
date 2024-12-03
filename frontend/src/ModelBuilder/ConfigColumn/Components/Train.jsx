import "../../ModelBuilder.css";

function Train({ trainInputs, handleInputChange }) {
    return (
        <div>
            <h2>Training Params</h2>
            <div>
                <label>Learning Rate: </label>
                <br/>
                <input
                    type="number"
                    value={trainInputs.lr}
                    onChange={(e) => handleInputChange("lr", e.target.value)}
                    className="input"
                />
            </div>
            <div>
                <label>Batch Size: </label>
                <br/>
                <input
                    type="number"
                    value={trainInputs.batch_size}
                    onChange={(e) => handleInputChange("batch_size", e.target.value)}
                    className="input"
                />
            </div>
            <div>
                <label>Epochs: </label>
                <br/>
                <input
                    type="number"
                    value={trainInputs.epochs}
                    onChange={(e) => handleInputChange("epochs", e.target.value)}
                    className="input"
                />
            </div>
            <br/>
        </div>
    );
}

export default Train;