import React, { useState, useEffect, useRef } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import Visualizer from "./Visualizer/Visualizer";
import ConfigColumn from "./ConfigColumn/ConfigColumn";
import "./ModelBuilder.css";

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
  const [backendResults, setBackendResults] = useState(null);
  const [isTraining, setIsTraining] = useState(false); // New state to track training status
  const intervalIdRef = useRef(null);
  const [trainInputs, setTrainInputs] = useState({
    lr: .01,
    batch_size: 64,
    epochs: 10,
  });

  const datasetItems = [
    { value: "MNIST", label: "MNIST" },
    { value: "CIFAR 10", label: "CIFAR 10" },
  ];

  const layerItems = [
    { value: "FcNN", label: "FcNN" },
    { value: "Conv", label: "Conv" },
  ];

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

  const createLayers = (newLayers) => {
    setLayers((prevLayers) => {
      const updatedLayers = [...prevLayers, ...newLayers];
      generateJson(updatedLayers);
      return updatedLayers;
    });
  };

  const generateJson = (updatedLayers) => {
    let datasetInputSize = 0;
    let datasetOutputSize = 0;
    if (selectedDataset === "MNIST") {
      datasetInputSize = 784;
      datasetOutputSize = 10;
    } else if (selectedDataset === "CIFAR 10") {
      datasetInputSize = 3072;
      datasetOutputSize = 10;
    }

    const modelBuilderJson = {
      model_config: {
        input: datasetInputSize,
        output: datasetOutputSize,
        dataset: selectedDataset,
        LR: trainInputs.lr,
        batch_size: trainInputs.batch_size,
        epochs: trainInputs.epochs,
        blocks: updatedLayers.map((layer) => ({
          block: layer.type || "FcNN",
          params: {
            output_size: layer.params.output_size,
            hidden_size: layer.params.hidden_size,
            num_hidden_layers: layer.params.num_hidden_layers,
          },
        })),
      },
      dataset: selectedDataset,
    };

    return modelBuilderJson;
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
      console.error("Error fetching logs:", error);
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
    const json = generateJson(layers);
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
  };

  const downloadModels = async () => {
    try {
        const token = await getAccessTokenSilently();
        const modelConfig = generateJson(layers);
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

  return (
    <div>
      <div className="container">
        <ConfigColumn
          selectedDataset={selectedDataset}
          datasetDropdownOpen={datasetDropdownOpen}
          toggleDatasetDropdown={toggleDatasetDropdown}
          datasetItems={datasetItems}
          handleDatasetClick={handleDatasetClick}
          layerItems={layerItems}
          createLayers={createLayers}
          removeLastBlock={removeLastBlock}
          layers={layers}
          trainInputs={trainInputs}
          setTrainInputs={setTrainInputs}
        />
        <Visualizer layers={layers} onLayerDragStop={onLayerDragStop} />
        <div className="buttons">
          <button
            className="sendBackend"
            onClick={handleSendJsonClick}
            disabled={isTraining}
          >
            {isTraining ? "Training in Progress..." : "TRAIN"}
          </button>
          <button
            className="stopTraining"
            onClick={stopTraining}
            disabled={!isTraining} //cant click unlesss traing is active
          >
            Stop Training
          </button>
          <button
              className="downloadModels"
              onClick={downloadModels}  // Remove the immediate invocation
              disabled={isTraining}
          >
              Download Models
          </button>
        </div>
        {backendResults && (
          <div className="backend-results">
            <h3>Backend Results:</h3>
            <pre>{JSON.stringify(backendResults, null, 2)}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelBuilder;
