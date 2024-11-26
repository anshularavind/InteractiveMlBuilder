import React, { useState } from "react";
import { useAuth0 } from '@auth0/auth0-react'
import ModelBuilder from "./ModelBuilder";
import DatasetSelection from "./DatasetSelection";
import Profile from "./Profile";
import TrainingControl from "./TrainingControl";


function App() {
  const [currentPage, setCurrentPage] = useState("start");
  const { loginWithRedirect, logout, user, isAuthenticated, isLoading} = useAuth0();

  const renderPage = () => {

    if (!isAuthenticated) {
      return (
        <div style={styles.container}>
          <h1 style={styles.header}>
            Welcome to the Interactive ML Model Builder
          </h1>
          <p style={styles.subtitle}>Please log in to get started:</p>
          <Button text="Login" onClick={() => loginWithRedirect()} />
        </div>
      );
    }

    switch (currentPage) {
      case "modelBuilder":
        return <ModelBuilder goBack={() => setCurrentPage("start")} />;
      case "datasetSelection":
        return <DatasetSelection goBack={() => setCurrentPage("start")} />;
      case "profile":
        return <Profile goBack={() => setCurrentPage("start")} user={user} />;
      case "trainingControl":
        return <TrainingControl goBack={() => setCurrentPage("start")} />;
      default:
        return (
          <div style={styles.container}>
            <h1 style={styles.header}>Welcome to the Interactive ML Model Builder</h1>
            <p style={styles.subtitle}>Select a feature to get started:</p>
            <Button text="Model Builder" onClick={() => setCurrentPage("modelBuilder")} />
            <Button text="Dataset Selection" onClick={() => setCurrentPage("datasetSelection")} />
            <Button text="Profile" onClick={() => setCurrentPage("profile")} />
            <Button text="Training Control" onClick={() => setCurrentPage("trainingControl")} />
            <Button text="Logout" onClick={() => logout({ returnTo: window.location.origin })} />
          </div>
        );
    }
  };

  return <div>{renderPage()}</div>;
}

const Button = ({ text, onClick }) => {
  const [hover, setHover] = useState(false);

  return (
    <button
      style={
        hover ? { ...styles.button, ...styles.buttonHover } : styles.button
      }
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={onClick}
    >
      {text}
    </button>
  );
};

const styles = {
  container: {
    textAlign: "center",
    padding: "20px",
    fontFamily: "'Poppins', sans-serif",
    backgroundColor: "#f0f8ff",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    position: "relative",
  },
  header: {
    fontSize: "2.5rem",
    color: "#4a90e2",
    marginBottom: "10px",
    fontWeight: 600,
  },
  subtitle: {
    fontSize: "1.2rem",
    color: "#555",
    marginBottom: "20px",
  },
  button: {
    fontSize: "1rem",
    margin: "10px",
    padding: "10px 20px",
    color: "#fff",
    backgroundColor: "#4a90e2",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    transition: "background-color 0.3s",
  },
  buttonHover: {
    backgroundColor: "#357abd",
  },
};

export default App;
