import React from "react";
import { BrowserRouter as Router, Route, Routes, Link, Navigate } from "react-router-dom";
import { useAuth0 } from '@auth0/auth0-react';
import ModelBuilder from "./ModelBuilder";
import About from "./About";
import Profile from "./Profile";
import './App.css'; // Import the CSS file

function App() {
  const { loginWithRedirect, logout, user, isAuthenticated, isLoading, error } = useAuth0();

  if (error) {
    console.log('error:', error);
    return <div className="container">Oops... {error.message}</div>;
  }

  if (isLoading) {
    return <div className="container">Loading...</div>;
  }

  return (
    <Router>
      <div className="fullScreen">
        {!isAuthenticated ? (
          <div className="container">
            <h1 className="header">Welcome to the Interactive ML Model Builder</h1>
            <p className="subtitle">Please log in to get started:</p>
            <Button text="Login" onClick={() => loginWithRedirect()} />
          </div>
        ) : (
          <>
            <nav className="nav">
              <div>
                <Link className="navLink" to="/home">Home</Link>
                <Link className="navLink" to="/model-builder">Model Builder</Link>
                <Link className="navLink" to="/about">About</Link>
              </div>
              <div className="userInfo">
                <Link to="/profile" className="userLink">
                  User: {user?.name}
                </Link>
                <button
                  className="logoutButton"
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
              <Route path="/about" element={<About />} />
              <Route path="/profile" element={<Profile user={user} />} />
            </Routes>
          </>
        )}
      </div>
    </Router>
  );
}

const Home = () => (
  <div className="fullScreenContent">
    <h1 className="header">Home Page for Interactive ML Model Builder</h1>
    <p className="subtitle">Select a feature to get started from the navigation bar above.</p>
  </div>
);

const Button = ({ text, onClick }) => {
  const [hover, setHover] = React.useState(false);

  return (
    <button
      className={hover ? "button buttonHover" : "button"}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      onClick={onClick}
    >
      {text}
    </button>
  );
};

export default App;
