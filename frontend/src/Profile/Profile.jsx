import React from "react";
import "./Profile.css";

function Profile({ user }) {
  const insights = [
    { label: "Models Trained", value: user.modelsTrained || 0 },
    { label: "Datasets Used", value: user.datasetsUsed || 0 },
    { label: "Layers Configured", value: user.layersConfigured || 0 },
    { label: "Recent Activity", value: user.recentActivities?.length || 0 },
  ];

  return (
    <div className="profile-container">
      <h1 className="profile-header">Welcome, {user.name || "User"}!</h1>
      <p className="profile-subtitle">Here are your ML Model Builder insights:</p>

      <div className="profile-insights">
        {insights.map((insight, index) => (
          <div key={index} className="insight-card">
            <h2 className="insight-value">{insight.value}</h2>
            <p className="insight-label">{insight.label}</p>
          </div>
        ))}
      </div>

      <div className="profile-activities">
        <h2>Recent Activities</h2>
        {user.recentActivities && user.recentActivities.length > 0 ? (
          <ul className="activities-list">
            {user.recentActivities.map((activity, index) => (
              <li key={index} className="activity-item">
                {activity}
              </li>
            ))}
          </ul>
        ) : (
          <p className="no-activities">No recent activities to display.</p>
        )}
      </div>
    </div>
  );
}

export default Profile;
