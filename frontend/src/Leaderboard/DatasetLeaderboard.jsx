import React, { useState, useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import SearchBar from "./SearchBar";


function DatasetLeaderboard() {
    const { getAccessTokenSilently } = useAuth0();
    const [datasets, setDatasets] = useState([]);
    const [searchTerm, setSearchTerm] = useState("");
    const [activeDatasetIndex, setActiveDatasetIndex] = useState(0);

    const getDatasets = async () => {
        try {
            const token = await getAccessTokenSilently();
            const response = await fetch('http://localhost:4000/api/datasets', {
                method: 'GET',
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${token}`,
                    Accept: "application/json",
                },
                credentials: "include",
                mode: "cors",
            });

            if (!response.ok) {
                throw new Error("Error fetching datasets");
            }
            const data = await response.json();
            console.log('Data:', data);
            if (data.datasets && Array.isArray(data.datasets)) {
                console.log('Datasets received:', data.datasets);
                return data.datasets;
            } else {
                throw new Error('Invalid data format for datasets.');
            }
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        getDatasets()
            .then(datasets => {
                setDatasets(datasets);
            })
            .catch(err => {
                console.error(err);
            });
    }, []);

    const handleSearch = (event) => {
        setSearchTerm(event.target.value.toLowerCase());
    };

    const activeDataset =
        datasets.length > 0 ? datasets[activeDatasetIndex] : null;

    // Filter models by search term
    const visualizeModel = async (model_uuid) => {
        // use get model_config route to get model config and then visualize
        let model_config = {};

        for (let i = 0; i < datasets.length; i++) {
            for (let j = 0; j < datasets[i].models.length; j++) {
                if (datasets[i].models[j].model_uuid === model_uuid) {
                    model_config = datasets[i].models[j];
                    break;
                }
            }
        }

        console.log('Model config:', JSON.stringify(model_config));

        

        // redirect to https://localhost:3000/model-builder
        window.location.href = 'https://localhost:3000/model-builder';

        sessionStorage.setItem("model_config", JSON.stringify(model_config));


    }

    const filteredModels =
        activeDataset && activeDataset.models
            ? activeDataset.models.filter(
                  (model) =>
                      String(model.username).toLowerCase().includes(searchTerm) ||
                      String(model.model_uuid).toLowerCase().includes(searchTerm)
              )
            : [];

    return (
        <div>
            <div className="dataset-tabs">
                {datasets.map((dataset, index) => (
                    <button
                        key={index}
                        className={`dataset-tab ${
                            activeDatasetIndex === index ? "active" : ""
                        }`}
                        onClick={() => setActiveDatasetIndex(index)}
                    >
                        {dataset.name}
                    </button>
                ))}
            </div>
            {activeDataset && (
                <div className="dataset-details">
                    <h3>{activeDataset.name}</h3>
                    <SearchBar
                        placeholder={`Search models in ${activeDataset.name}`}
                        value={searchTerm}
                        onChange={handleSearch}
                        totalUsers={activeDataset.models.length}
                        matchedUsers={filteredModels.length}
                    />
                    <table>
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Username</th>
                                <th>Model UUID</th>
                                <th>{activeDataset.metric}</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filteredModels.map((model, index) => (
                                <tr key={index}>
                                    <td>{index + 1}</td>
                                    <td>{model.username}</td>
                                    <td>
                                   
                                        <button 
                                        className="UUID_button"
                                        onClick={() => visualizeModel(model.model_uuid)}>
                                            {model.model_uuid}
                                        </button>
                                       
                                    </td>

                                    <td>{model.metric}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

export default DatasetLeaderboard;