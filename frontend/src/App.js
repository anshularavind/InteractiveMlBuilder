import React from "react";
import { BrowserRouter as Router, Route, Routes, Link, Navigate } from "react-router-dom";
import { useAuth0 } from '@auth0/auth0-react';
import ModelBuilder from "./ModelBuilder";
import About from "./About";
import Profile from "./Profile";
import TrainingControl from "./TrainingControl";

function App() {
  const { loginWithRedirect, logout, user, isAuthenticated, isLoading, error } = useAuth0();

  if (error) {
    console.log('error:', error);
    return <div style={styles.container}>Oops... {error.message}</div>;
  }

  if (isLoading) {
    return <div style={styles.container}>Loading...</div>;
  }

  return (
    <Router>
      <div style={styles.fullScreen}>
        {!isAuthenticated ? (
          <div style={styles.container}>
            <h1 style={styles.header}>
              Welcome to the Interactive ML Model Builder
            </h1>
            <p style={styles.subtitle}>Please log in to get started:</p>
            <Button text="Login" onClick={() => loginWithRedirect()} />
          </div>
        ) : (
          <>
            <nav style={styles.nav}>
              <div>
                <Link style={styles.navLink} to="/home">Home</Link>
                <Link style={styles.navLink} to="/model-builder">Model Builder</Link>
                <Link style={styles.navLink} to="/About">About </Link>
                <Link style={styles.navLink} to="/profile">Profile</Link>
               {/*<Link style={styles.navLink} to="/training-control">Training Control</Link>*/}
              </div>
              <div style={styles.userInfo}>
                User: {user?.name}
                <button
                  style={styles.logoutButton}
                  onClick={() => logout({ returnTo: window.location.origin })}
                >
                  Logout
                </button>
              </div>
            </nav>
            <Routes>
              <Route path="/" element={<Navigate to="/home" />} />
              <Route path="/home" element={<Home />} />
              <Route path="/model-builder" element={<ModelBuilder />} />
              <Route path="/About" element={<About />} />
              <Route path="/profile" element={<Profile user={user} />} />
              {/*<Route path="/training-control" element={<TrainingControl />} />*/}
            </Routes>
          </>
        )}
      </div>
    </Router>
  );
}

const Home = () => (
  <div style={styles.fullScreenContent}>
    <h1 style={styles.header}>Home Page for Interactive ML Model Builder</h1>
    <p style={styles.subtitle}>Select a feature to get started from the navigation bar above.</p>
  </div>
);

const Button = ({ text, onClick }) => {
  const [hover, setHover] = React.useState(false);

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
  fullScreen: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    margin: 0,
  },
  nav: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "#f0f8ff",
    padding: "10px 20px",
    borderBottom: "1px solid #ccc",
  },
  navLink: {
    margin: "0 10px",
    textDecoration: "none",
    color: "#4a90e2",
    fontSize: "1.1rem",
    fontWeight: "bold",
  },
  userInfo: {
    display: "flex",
    alignItems: "center",
    fontSize: "1rem",
    color: "#555",
  },
  logoutButton: {
    marginLeft: "10px",
    padding: "5px 10px",
    fontSize: "0.9rem",
    backgroundColor: "#4a90e2",
    color: "#fff",
    border: "none",
    borderRadius: "4px",
    cursor: "pointer",
  },
  fullScreenContent: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#f0f8ff",
  },
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
