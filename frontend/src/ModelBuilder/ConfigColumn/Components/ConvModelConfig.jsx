function ConvModelConfig({ blockInputs, handleInputChange, createLayers, selectedDataset }) {
    const is2D = selectedDataset === 'MNIST' || selectedDataset === 'CIFAR10';

    return (
        <div>
            <div>
                <label>Kernel Size: </label>
                <br/>
                <input
                    type='number'
                    value={blockInputs.kernelsSize}
                    onChange={(e) => handleInputChange('kernelSize', e.target.value)}
                    className='input'
                />
            </div>
            <div>
                <label>Output Channels: </label>
                <br/>
                <input
                    type='number'
                    value={blockInputs.numKernels}
                    onChange={(e) => handleInputChange('numKernels', e.target.value)}
                    className='input'
                />
            </div>
            <div>
                <label>{is2D ? 'Pooling Output Side-Length' : 'Pooling Output Size'}: </label>
                <br/>
                <input
                    type='number'
                    value={blockInputs.outputSize}
                    onChange={(e) => handleInputChange('outputSize', e.target.value)}
                    className='input'
                />
            </div>
            <button className='addButton' onClick={createLayers}>Add Block</button>
        </div>

    );
}

export default ConvModelConfig;