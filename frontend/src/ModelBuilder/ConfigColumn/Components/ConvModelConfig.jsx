import "../../ModelBuilder.css";

function ConvModelConfig({ blockInputs, handleInputChange, createLayers }) {
    return (
        <div>
            <h4>Configure Block Parameters</h4>
            <div>
                <label>Kernel Size: </label>
                <input
                    type='number'
                    value={blockInputs.kernelsSize}
                    onChange={(e) => handleInputChange('kernelSize', e.target.value)}
                    className='input'
                />
            </div>
            <div>
                <label>Output Channels: </label>
                <input
                    type='number'
                    value={blockInputs.numKernels}
                    onChange={(e) => handleInputChange('numKernels', e.target.value)}
                    className='input'
                />
            </div>
            <button className='addButton' onClick={createLayers}>Add Block</button>
        </div>
    );
}

export default ConvModelConfig;