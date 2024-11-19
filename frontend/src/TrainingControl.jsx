import React from "react";

function TrainingControl({ goBack }) {
  return (
    <div style={styles.container}>
      <button style={styles.goBackButton} onClick={goBack}>
        Back to Home
      </button>
      <h2 style={styles.header}>Training Control Page</h2>
    </div>
  );
}

const styles = {
  container: {
    textAlign: "center",
    padding: "20px",
    fontFamily: "'Poppins', sans-serif",
    backgroundColor: "#f8f9fa",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
  },
  goBackButton: {
    position: "absolute",
    top: "20px",
    left: "20px",
    fontSize: "1rem",
    padding: "10px 15px",
    color: "#fff",
    backgroundColor: "#4a90e2",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
  },
  header: {
    fontSize: "2rem",
    color: "#4a90e2",
  },
};

export default TrainingControl;
