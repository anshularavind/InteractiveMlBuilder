import React, { useState, useEffect } from "react";
import "./Profile.css";
import ModelsTable from "../Models/Models";
import { useAuth0 } from "@auth0/auth0-react";

function Profile() {
  const {
    getAccessTokenSilently,
    user,
    isAuthenticated,
    isLoading,
  } = useAuth0();

  const [userInfo, setUserInfo] = useState(null); // Store the current user's data
  const [error, setError] = useState(null);

  useEffect(() => {
    const getUserInfo = async () => {
      try {
        const token = await getAccessTokenSilently();
        console.log("Fetching users from backend...");
        const response = await fetch("http://localhost:4000/api/user-info", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
            Accept: "application/json",
          },
          credentials: "include",
          mode: "cors",
        });
  
        if (!response.ok) {
          throw new Error(`Error fetching users: ${response.statusText}`);
        }
        const data = await response.json();
        if (data.user_info) {
          console.log("Users received:", data.user_info);
          setUserInfo(data.user_info); // Update state with user data
        } else {
          console.error("Invalid data format for users.");
          setUserInfo({}); // Set to an empty object if data is invalid
        }
      } catch (err) {
        console.error(err);
        setError("Failed to fetch user insights");
        setUserInfo({}); // Ensure state is updated to stop the loading spinner
      }
      const token = await getAccessTokenSilently();
  console.log("Token fetched:", token);
    };
  
    getUserInfo();
  }, [getAccessTokenSilently]);
  
  

  if (isLoading || !userInfo) {
    return <div>Loading user insights...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (Object.keys(userInfo).length === 0) {
    return <div>No data available for this user.</div>;
  }

  if (!userInfo || Object.keys(userInfo).length === 0) {
    return <div>No data available for this user.</div>;
  }
  
  const insights = [
    { label: "Models Trained", value: userInfo.num_models || 0 },
    { label: "Datasets Used", value: userInfo.num_datasets || 0 },
    { label: "Highest Rank", value: userInfo.highest_rank || 0 },
  ];

  return (
    <div className="profile-container">
      <h1 className="profile-header">Welcome, {user.name || "User"}!</h1>
      <p className="profile-subtitle">Here are your ML Model Builder insights:</p>

      <div className="profile-insights">
      {insights.map((insight, index) => (
        insight.label === "Models Trained" ? (
          <div key={index} className="insight-card">
            <h2 className="insight-value">{insight.value}</h2>
            <p className="insight-label">{insight.label}</p>
          </div>
        ) : (
          <div key={index} className="insight-card">
            <h2 className="insight-value">{insight.value}</h2>
            <p className="insight-label">{insight.label}</p>
          </div>
        )
      ))}
      <ModelsTable />
      </div>
    </div>
  );
}

export default Profile;
