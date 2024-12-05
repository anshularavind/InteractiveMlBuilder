import React from 'react';
import { useAuth0 } from '@auth0/auth0-react';
import { useState, useEffect, useCallback} from 'react';
import { data } from 'react-router-dom';
import '../Leaderboard/Leaderboard.css';

const ModelsTable = () => {
    const { getAccessTokenSilently } = useAuth0();
    const [models, setModels] = useState([]); // Initialize with empty array

    const getModels = async () => {
        try {
            const token = await getAccessTokenSilently();
            const response = await fetch('http://127.0.0.1:4000/api/models', {
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
                throw new Error("Error fetching models");
            }
            const data = await response.json();
            if (data.models && Array.isArray(data.models)) {
                console.log('Models received:', data.models);
                return data.models;
            } else {
                throw new Error('Invalid data format for models.');
            }
        } catch (error) {
            console.error(error);
        }
    };

    useEffect(() => {
        getModels()
            .then(models => {
                setModels(models);
            })
            .catch(err => {
                console.error(err);
            });
    }, []);

    // // Add conditional rendering
    // if (!models.length) {
    //     return <div>Loading...</div>;
    // }

    const handleLookAtModel = async (model_config) => {

        console.log('Model config:', JSON.stringify(model_config));

        

        // redirect to https://localhost:3000/model-builder
        window.location.href = 'https://localhost:3000/model-builder';

        sessionStorage.setItem("model_config", JSON.stringify(model_config));
    };

    return (
        <table>
            <thead>
                <tr>
                    <th>Model UUID</th>
                    <th>Action</th>
                    <th>Dataset</th>
                </tr>
            </thead>
            <tbody>
                {models.map((model) => (
                    <tr key={model.model_uuid}>
                        <td>{model.model_uuid}</td>
                        <td>
                            <button className="UUID_button" onClick={() => handleLookAtModel(model)}>Look at Model</button>
                        </td>
                        <td>{ model.model_config.dataset }</td>
                    </tr>
                ))}
            </tbody>
        </table>
    );
};

export default ModelsTable;