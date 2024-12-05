import React, { useState } from "react";
import "./Leaderboard.css";
import UserLeaderboard from "./UserLeaderboard";
import DatasetLeaderboard from "./DatasetLeaderboard";

function Leaderboard() {
    const [activeTab, setActiveTab] = useState("users");

    return (
        <div className="leaderboard-container">
            <div className="main-tabs">
                <button
                    className={`main-tab ${activeTab === "users" ? "active" : ""}`}
                    onClick={() => setActiveTab("users")}
                >
                    Users
                </button>
                <button
                    className={`main-tab ${activeTab === "datasets" ? "active" : ""}`}
                    onClick={() => setActiveTab("datasets")}
                >
                    Datasets
                </button>
            </div>
            <h2 className="leaderboard-title">LEADERBOARD</h2>
            <div className="leaderboard-content">
                {activeTab === "users" && <UserLeaderboard />}
                {activeTab === "datasets" && <DatasetLeaderboard />}
            </div>
        </div>
    );
}

export default Leaderboard;