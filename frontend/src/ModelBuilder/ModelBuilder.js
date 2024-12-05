import React, { useState, useEffect, useRef } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css";
import GraphParser from "./GraphParser";
import TrainingGraph from "./TrainingGraph";

function ModelBuilder() {
  const {
    getAccessTokenSilently,
    loginWithRedirect,
    logout,
    user,
    isAuthenticated,
    isLoading,
  } = useAuth0();
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [datasetDropdownOpen, setDatasetDropdownOpen] = useState(false);
  const [selectedLayer, setSelectedLayer] = useState(null);
  const [layerDropdownOpen, setLayerDropdownOpen] = useState(false);
  const [layers, setLayers] = useState([]);
  const [modelConfig, setModelConfig] = useState(null);
  const [blockCount, setBlockCount] = useState(0);
  const [backendResults, setBackendResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false); // New state to track training status
  const intervalIdRef = useRef(null);
  const [trainInputs, setTrainInputs] = useState({
    lr: .01,
    batch_size: 64,
    epochs: 10,
  });

  // New state to track training status
  const [graphData, setGraphData] = useState({
    epochs: [],
    losses: [],
    accuracies: [],
  });
  const [errorMessage, setErrorMessage] = useState(null);

  const datasetItems = [
    { value: "MNIST", label: "MNIST" },
    { value: "CIFAR10", label: "CIFAR10" },
    { value: "AirQuality", label: "AirQuality" },
    { value: "ETTh1", label: "ETTh1" },
  ];

  const datasetSizesMap = {
    "MNIST": { inputSize: 784, outputSize: 10 },
    "CIFAR10": { inputSize: 3072, outputSize: 10 },
    "AirQuality": { inputSize: 11, outputSize: 2 },
    "ETTh1": { inputSize: 9, outputSize: 3 }
  };

  const handleDatasetClick = (item) => {
    setSelectedDataset(item.value);
    setDatasetDropdownOpen(false);
  };

  const toggleDatasetDropdown = () => {
    setDatasetDropdownOpen(!datasetDropdownOpen);
  };

  const handleLayerClick = (item) => {
    setSelectedLayer(item.value);
    setLayerDropdownOpen(false);
  };

  const toggleLayerDropdown = () => {
    setLayerDropdownOpen(!layerDropdownOpen);
  };

  const loadModelConfig = () => {
    let json = sessionStorage.getItem("model_config");
    if (json !== null) {
      if ("model_config" in JSON.parse(json))
        return JSON.parse(json)["model_config"];
    }
    return null;
  }

  const createLayers = (newLayers) => {
    setLayers((prevLayers) => [...prevLayers, ...newLayers]);
  };

  useEffect(() => {
    const newConfig = generateJson(layers);
    console.log(layers);
    setModelConfig(newConfig);
    if (newConfig?.model_config?.blocks?.length > 0)
      sessionStorage.setItem("model_config", JSON.stringify(newConfig));
  }, [layers]);

  const generateJson = (updatedLayers) => {
    let datasetInputSize = datasetSizesMap[selectedDataset]?.inputSize ?? 0;
    let datasetOutputSize = datasetSizesMap[selectedDataset]?.outputSize ?? 0;
    console.log(updatedLayers);

    let blocks = updatedLayers.map((layer) => {
      if (layer.type === 'FcNN' || !layer.type) {
        return [{
          block: 'FcNN',
          params: {
            output_size: layer.params.output_size,
            hidden_size: layer.params.hidden_size,
            num_hidden_layers: layer.params.num_hidden_layers,
          },
        }];
      } else if (layer.type === 'Conv') {
        return [
          {
            block: 'Conv',
            params: {
              num_kernels: layer.params.num_kernels,
              kernel_size: layer.params.kernel_size,
              stride: layer.params.stride,
              padding: layer.params.padding,
              output_size: layer.params.output_size,
            },
          },
          {
            block: 'Pool',
            params: {
              output_size: layer.params.output_size,
            }
          },
        ];
      }
    });

    return {
      model_config: {
        input: datasetInputSize,
        output: datasetOutputSize,
        dataset: selectedDataset,
        LR: trainInputs.lr,
        batch_size: trainInputs.batch_size,
        epochs: trainInputs.epochs,
        blocks: blocks.flat(),  // unpack nested arrays of blocks
      },
    };
  };

  const sendJsonToBackend = async (json) => {
    try {
      clearInterval(intervalIdRef.current); // Stop fetching logs
      const token = await getAccessTokenSilently();

      const response = await fetch("http://127.0.0.1:4000/api/define-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
          Accept: "application/json",
        },
        credentials: "include",
        mode: "cors",
        body: JSON.stringify(json),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error sending JSON to backend:", error);
      throw error;
    }
  };

  const startTraining = async (json) => {
    try {
      const token = await getAccessTokenSilently();

      const response = await fetch("http://127.0.0.1:4000/api/train", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
          Accept: "application/json",
        },
        credentials: "include",
        mode: "cors",
        body: JSON.stringify(json),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("Error starting training:", error);
      throw error;
    }
  };

  const fetchLogs = async (json) => {
    try {
      const token = await getAccessTokenSilently();
      const response = await fetch("http://127.0.0.1:4000/api/train-logs", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
          Accept: "application/json",
        },
        credentials: "include",
        mode: "cors",
        body: JSON.stringify(json),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      setBackendResults(data);

      if (data.logs) {
        const newLogs = data.logs; // Replace with actual log structure
        setGraphData((prevData) => ({
          epochs: [...prevData.epochs, ...newLogs.map((log) => log.epoch)],
          losses: [...prevData.losses, ...newLogs.map((log) => log.loss)],
          accuracies: [
            ...prevData.accuracies,
            ...newLogs.map((log) => log.accuracy),
          ],
        }));
      }

      // If there is an error or the logs indicate training is complete, stop fetching logs
      if (data.error !== null && data.error !== "") {
        clearInterval(intervalIdRef.current);
        setIsTraining(false);
      }

      // If final log message is received, stop fetching logs
      if (data.output) {
        let output_data = data.output;
        output_data = output_data.replace(/\n+$/, "");  // Remove trailing newlines
        let output_lines = output_data.split("\n");
        if (output_lines.length > 0 && output_lines[output_lines.length - 1].startsWith("Final")) {
          clearInterval(intervalIdRef.current);
          setIsTraining(false); // Stop training when logs indicate completion
        }
      }
    } catch (error) {
      setErrorMessage("Error fetching logs. Please try again.");
      console.error("Error fetching logs:", error);
      clearInterval(intervalIdRef.current);
      setIsTraining(false);
    }
  };

  const startFetchingLogs = (json) => {
    intervalIdRef.current = setInterval(() => fetchLogs(json), 1000);
  };

  const stopTraining = async () => {
    try {
      const token = await getAccessTokenSilently();

      const response = await fetch("http://127.0.0.1:4000/api/stop-training", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
          Accept: "application/json",
        },
        credentials: "include",
        mode: "cors",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      setIsTraining(false);
      clearInterval(intervalIdRef.current); // Stop fetching logs
      console.log("Training stopped successfully.");
    } catch (error) {
      console.error("Error stopping training:", error);
    }
  };

  const handleSendJsonClick = async () => {
    let json = JSON.parse(JSON.stringify(modelConfig));
    await sendJsonToBackend(json);
    await startTraining(json);
    setIsTraining(true); // Set training state to true
    startFetchingLogs(json);
  };

  const onLayerDragStop = (id, data) => {
    setLayers((prevLayers) =>
      prevLayers.map((layer) =>
        layer.id === id ? { ...layer, position: { x: data.x, y: data.y } } : layer
      )
    );
  };

  const removeLastBlock = () => {
    setLayers((prevLayers) => {
      if (prevLayers.length === 0) return prevLayers;
      return prevLayers.slice(0, -1);
    });

    setBlockCount((prevCount) => (prevCount > 0 ? prevCount - 1 : 0));
    sessionStorage.setItem("model_config", "{}");
  };

  const removeBlocks = () => {
    setLayers([]);
    setBlockCount(0);
    sessionStorage.setItem("model_config", "{}");
  };

  const downloadModels = async () => {
    try {
        const token = await getAccessTokenSilently();
        const response = await fetch("http://127.0.0.1:4000/api/download-model", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify(modelConfig),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `model_${Date.now()}.pt`;
        document.body.appendChild(a);
        a.click();
        
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        console.error("Error downloading model:", error);
    }
  };

  const isTrainingDisabled = () => {
    if (isTraining || layers.length === 0)
      return true;
    let datasetOutputSize = datasetSizesMap[selectedDataset]?.outputSize ?? 0;
    let lastLayerOutputSize = layers[layers.length - 1].params.output_size;
    if (datasetOutputSize === 0 || lastLayerOutputSize !== datasetOutputSize)
        return true;
    return false;
  }

  return (
    <div>
      <div className="container">
        <ConfigColumn
          selectedDataset={selectedDataset}
          setSelectedDataset={setSelectedDataset}
          datasetSizesMap={datasetSizesMap}
          datasetDropdownOpen={datasetDropdownOpen}
          toggleDatasetDropdown={toggleDatasetDropdown}
          datasetItems={datasetItems}
          handleDatasetClick={handleDatasetClick}
          blockCount={blockCount}
          setBlockCount={setBlockCount}
          createLayers={createLayers}
          removeLastBlock={removeLastBlock}
          layers={layers}
          trainInputs={trainInputs}
          setTrainInputs={setTrainInputs}
          loadModelConfig={loadModelConfig}
        />
        <Visualizer layers={layers} onLayerDragStop={onLayerDragStop} />
        <div className="buttons">
          <button
              className="sendBackend"
              onClick={handleSendJsonClick}
              disabled={isTrainingDisabled()}
          >
            {isTraining ? "Training in Progress..." : "Train"}
          </button>
          <button
              className="stopTraining"
              onClick={stopTraining}
              disabled={!isTraining} //cant click unlesss traing is active
          >
            Stop Training
          </button>
          <button className="downloadModels" onClick={downloadModels} disabled={isTraining}>
            Download Models
          </button>
          <button className="clearButton" onClick={removeBlocks}>Clear Model</button>
        </div>
        <div>
          <div className="training-graphs">
            <GraphParser backendResults={backendResults}/>
          </div>
          {backendResults && (
            <div className="backend-results">
              <h3>Results</h3>
              {backendResults.error && (
                <div className="error-message">
                  <h3>Error:</h3>
                  <p>{backendResults.error}</p>
                </div>
              )}
              <pre>LOSSES: {backendResults.loss}</pre>
              <pre>OUTPUTS: {backendResults.output}</pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}



export default ModelBuilder;
