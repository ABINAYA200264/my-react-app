import React, { useState } from "react";

function UserInputForm({ onSubmit }) {
  const [videoUrl, setVideoUrl] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [supervisor, setSupervisor] = useState("");
  const [vehicleNumber, setVehicleNumber] = useState("");
  const [customSupervisor, setCustomSupervisor] = useState("");
  const [customVehicleNumber, setCustomVehicleNumber] = useState("");

// Removed duplicate and invalid top-level fetch logic.

  const supervisorOptions = ["Supervisor 1", "Supervisor 2", "Supervisor 3", "Other"];
  const vehicleOptions = ["Vehicle A123", "Vehicle B456", "Vehicle C789", "Other"];

  const handleSubmit = async (e) => {
    e.preventDefault();

    const finalSupervisor = supervisor === "Other" ? customSupervisor : supervisor;
    const finalVehicleNumber = vehicleNumber === "Other" ? customVehicleNumber : vehicleNumber;

    try {
      let body;
      let headers;

      if (videoFile) {
        // Prepare multipart/form-data with file upload
        body = new FormData();
        body.append("file", videoFile);
        body.append("supervisor", finalSupervisor);
        body.append("vehicle_number", finalVehicleNumber);
        headers = {}; // Let browser set Content-Type including boundary
      } else {
        // JSON payload with video URL
        body = JSON.stringify({
          video_url: videoUrl,
          supervisor: finalSupervisor,
          vehicle_number: finalVehicleNumber,
        });
        headers = { "Content-Type": "application/json" };
      }

      const resp = await fetch("http://localhost:8000/submit-info", {
        method: "POST",
        headers,
        body,
      });

      if (resp.ok) {
        onSubmit({ videoUrl, videoFile, supervisor: finalSupervisor, vehicleNumber: finalVehicleNumber });
      } else {
        alert("Failed to submit info");
      }
    } catch (error) {
      alert("Network error");
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Enter Video and Vehicle Details</h2>
      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
          placeholder="Video URL"
          required={!videoFile} // URL required only if file not selected
          style={styles.input}
        />

        <label style={styles.label}>Or Upload Video File</label>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => {
            setVideoFile(e.target.files[0]);
            if (e.target.files) setVideoUrl(""); // Clear URL if file chosen
          }}
          style={{ marginBottom: 16 }}
        />

        <label style={styles.label}>Supervisor</label>
        <select
          value={supervisor}
          onChange={(e) => setSupervisor(e.target.value)}
          required
          style={styles.select}
        >
          <option value="" disabled>
            Select supervisor
          </option>
          {supervisorOptions.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        {supervisor === "Other" && (
          <input
            type="text"
            placeholder="Enter supervisor name"
            value={customSupervisor}
            onChange={(e) => setCustomSupervisor(e.target.value)}
            required
            style={styles.input}
          />
        )}

        <label style={styles.label}>Vehicle Number</label>
        <select
          value={vehicleNumber}
          onChange={(e) => setVehicleNumber(e.target.value)}
          required
          style={styles.select}
        >
          <option value="" disabled>
            Select vehicle number
          </option>
          {vehicleOptions.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        {vehicleNumber === "Other" && (
          <input
            type="text"
            placeholder="Enter vehicle number"
            value={customVehicleNumber}
            onChange={(e) => setCustomVehicleNumber(e.target.value)}
            required
            style={styles.input}
          />
        )}

        <button type="submit" style={styles.button}>
          Submit
        </button>
      </form>
    </div>
  );
}

const styles = {
  container: {
    maxWidth: 400,
    margin: "80px auto",
    padding: 30,
    backgroundColor: "#f0f8ff",
    borderRadius: 12,
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    textAlign: "center",
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
  },
  title: {
    marginBottom: 24,
    color: "#007acc",
  },
  form: {
    display: "flex",
    flexDirection: "column",
  },
  label: {
    textAlign: "left",
    marginBottom: 6,
    fontWeight: "bold",
    color: "#007acc",
  },
  input: {
    marginBottom: 16,
    padding: "12px 15px",
    fontSize: 16,
    borderRadius: 6,
    border: "1px solid #007acc",
    outline: "none",
    transition: "border-color 0.3s ease",
  },
  select: {
    marginBottom: 16,
    padding: "12px 15px",
    fontSize: 16,
    borderRadius: 6,
    border: "1px solid #007acc",
    outline: "none",
    backgroundColor: "white",
    transition: "border-color 0.3s ease",
  },
  button: {
    padding: 14,
    fontSize: 18,
    backgroundColor: "#007acc",
    color: "white",
    border: "none",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: "bold",
    transition: "background-color 0.3s ease",
  },
};

export default UserInputForm;
