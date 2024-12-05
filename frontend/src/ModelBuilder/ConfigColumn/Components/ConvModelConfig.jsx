

function ConvModelConfig({ blockInputs, handleInputChange, createLayers, selectedDataset }) {
    const is2D = selectedDataset === 'MNIST' || selectedDataset === 'CIFAR10';

    return (
        <div className="convConfig">
            <div className="convInputGroup">
                <label>Kernel Size:</label>
                <input
                    type="number"
                    value={blockInputs.kernelSize}
                    onChange={(e) => handleInputChange('kernelSize', e.target.value)}
                    className="convInput"
                />
            </div>
            <div className="convInputGroup">
                <label>Output Channels:</label>
                <input
                    type="number"
                    value={blockInputs.numKernels}
                    onChange={(e) => handleInputChange('numKernels', e.target.value)}
                    className="convInput"
                />
            </div>
            <div className="convInputGroup">
                <label>{is2D ? 'Pooling Output Side-Length' : 'Pooling Output Size'}:</label>
                <input
                    type="number"
                    value={blockInputs.outputSize}
                    onChange={(e) => handleInputChange('outputSize', e.target.value)}
                    className="convInput"
                />
            </div>
            <button className="convAddButton" onClick={createLayers}>
                Add Block
            </button>
        </div>
    );
}

export default ConvModelConfig;
