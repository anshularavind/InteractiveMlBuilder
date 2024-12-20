import React from "react";
import { BrowserRouter as Router, Route, Routes, Link, Navigate } from "react-router-dom";
import { Button } from 'react-bootstrap';
import { useAuth0 } from '@auth0/auth0-react';
import ModelBuilder from "../ModelBuilder/ModelBuilder";
import About from "../About/About";
import Profile from "../Profile/Profile";
import Home from "../Home/Home";
import Leaderboard from "../Leaderboard/Leaderboard";
import Models from "../Models/Models";
import './App.css';
// App.js

function App() {

  const { user, isAuthenticated, isLoading, error, loginWithRedirect, logout, getAccessTokenSilently } = useAuth0();  
  const getToken = async () => {
    return await getAccessTokenSilently({
      audience: 'https://InteractiveMlApi',
      scope: 'read:current_user',
      authorizationParams: {
        response_type: 'token id_token',
        token_type: 'JWT'
      }
    });
  };

  getToken()
  .then(accessToken => {
    console.log('Access Token:', accessToken);
  })
  .catch(err => {
    console.error('Error getting access token:', err);
  });

  if (error) {
    console.log('error:', error);
    return <div className="container">Oops... {error.message}</div>;
  }


  if (isLoading) {
    return <div className="container">Loading...</div>;
  }

  // define a function to clear the local storage and log out the user
  const handleLogout = () => {
    sessionStorage.clear();
    logout({ returnTo: window.location.origin });
  };

  return (

    <Router>
      <div className="fullscreen">
        {!isAuthenticated ? (
          <div className="mainContainer">
          <h1 className="header">
            Welcome to the Interactive ML Model Builder
            </h1>
          <p className="subtitle">
            Please log in to get started:
            </p>
          <div className="buttonContainer">
          <Button className="loginButton" onClick={() => loginWithRedirect()}>
            Login
          </Button>
          </div>
        </div>
        ) : (
          <>
            <nav className="nav">
              <div>
                <Link className="navLink" to="/home">Home</Link>
                <Link className="navLink" to="/model-builder">Model Builder</Link>
                <Link className="navLink" to="/about">About</Link>
                <Link className="navLink" to="/leaderboard">Leaderboard</Link>
              </div>
              <div className="userInfo">
                <Link to="/profile" className="userLink">
                  User: {user?.name}
                </Link>
                <button
                  className="logoutButton"
                  onClick={() => handleLogout()}
                >
                  Logout
                </button>
              </div>
            </nav>
            <Routes>
              <Route path="/" element={<Navigate to="/home" />} />
              <Route path="/home" element={<Home />} />
              <Route path="/model-builder" element={<ModelBuilder/>} />
              <Route path="/about" element={<About />} />
              <Route path="/profile" element={<Profile user={user} />} />
              <Route path="/leaderboard" element={<Leaderboard />} />
              <Route path="/models" element={<Models/>} />
            </Routes>
          </>
        )}
      </div>
    </Router>
  );
}


export default App;
