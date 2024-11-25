import React, { useState, useEffect } from "react";
import ModelBuilder from "./ModelBuilder";
import DatasetSelection from "./DatasetSelection";
import Profile from "./Profile";
import TrainingControl from "./TrainingControl";

function App() {
  const [currentPage, setCurrentPage] = useState("start");
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const login = () => {
    window.location.href = "http://localhost:3000/login";
  };

  const logout = () => {
    window.location.href = "http://localhost:3000/logout";
  };

  useEffect(() => {
    const fetchSession = async () => {
      try {
        const response = await fetch("http://localhost:3000/session", {
          credentials: "include",
        });

        if (!response.ok) {
          throw new Error("Failed to fetch session data");
        }

        const data = await response.json();
        setUser(data.user || null);
      } catch (error) {
        console.error("Error fetching session:", error);
        setUser(null);
      } finally {
        setIsLoading(false);
      }
    };

    fetchSession();
  }, []);

  const renderPage = () => {
    if (isLoading) {
      return (
        <div style={styles.container}>
          <h1 style={styles.header}>Loading...</h1>
        </div>
      );
    }

    if (!user) {
      return (
        <div style={styles.container}>
          <h1 style={styles.header}>Welcome to the Interactive ML Model Builder</h1>
          <p style={styles.subtitle}>Please log in to get started:</p>
          <Button text="Login" onClick={login} />
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
            <Button
              text="Model Builder"
              onClick={() => setCurrentPage("modelBuilder")}
            />
            <Button
              text="Dataset Selection"
              onClick={() => setCurrentPage("datasetSelection")}
            />
            <Button text="Profile" onClick={() => setCurrentPage("profile")} />
            <Button
              text="Training Control"
              onClick={() => setCurrentPage("trainingControl")}
            />
            <Button text="Logout" onClick={logout} />
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
