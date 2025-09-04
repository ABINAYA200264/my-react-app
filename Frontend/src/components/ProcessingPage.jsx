import React, { useState, useEffect, useRef } from "react";
import ReactHlsPlayer from 'react-hls-player';

function ProcessingPage() {
  const [videoPath, setVideoPath] = useState("");
  const [supervisor, setSupervisor] = useState("");
  const [vehicleNumber, setVehicleNumber] = useState("");
  const [customSupervisor, setCustomSupervisor] = useState("");
  const [customVehicleNumber, setCustomVehicleNumber] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);

  // Detect backend URL
  const API_URL = process.env.REACT_APP_BACKEND_URL || window.location.origin;

  const [startHover, setStartHover] = useState(false);
  const [stopHover, setStopHover] = useState(false);
  const pollingIntervalRef = useRef(null);

  const supervisorOptions = ["Supervisor 1", "Supervisor 2", "Supervisor 3", "Other"];
  const vehicleOptions = ["Vehicle A123", "Vehicle B456", "Vehicle C789", "Other"];
  const modelOptions = ["Single Box", "Multiple Box", "4_5_6 Box"];

  const finalSupervisor = supervisor === "Other" ? customSupervisor : supervisor;
  const finalVehicleNumber = vehicleNumber === "Other" ? customVehicleNumber : vehicleNumber;

  const startProcessing = async () => {
    if (!videoPath) {
      alert("Please enter the full local video file path or live stream URL");
      return;
    }
    if (!supervisor || !vehicleNumber || !selectedModel) {
      alert("Please fill all fields before starting processing");
      return;
    }

    setProcessing(true);
    setResult(null);

    const formData = new FormData();
    formData.append("video_url", videoPath);
    formData.append("supervisor_name", finalSupervisor);
    formData.append("vehicle_no", finalVehicleNumber);
    formData.append("selected_model", selectedModel);

    try {
      const response = await fetch(`${API_URL}/process-video`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json();
        alert(`Error starting processing: ${err.detail || "Unknown error"}`);
        setProcessing(false);
        return;
      }

      startPolling();
    } catch (error) {
      alert("Failed to start processing: " + error.message);
      setProcessing(false);
    }
  };

  const startPolling = () => {
    if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/processing-result`);
        const data = await res.json();
        if (data.result) setResult(data.result);
        if (data.status === "done") {
          setProcessing(false);
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      } catch (error) {
        console.error("Error polling processing result:", error);
      }
    }, 2000);
  };

  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
    };
  }, []);

  const stopProcessing = async () => {
    try {
      await fetch(`${API_URL}/stop-processing`, { method: "POST" });
      setProcessing(false);
      if (pollingIntervalRef.current) clearInterval(pollingIntervalRef.current);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}> 🎥 Video Detection with Bounding Boxes</h2>

      <input
        type="text"
        placeholder="Paste local video path or live stream URL"
        value={videoPath}
        onChange={(e) => setVideoPath(e.target.value)}
        style={styles.input}
      />

      <label style={styles.label}>Supervisor</label>
      <select
        style={styles.select}
        value={supervisor}
        onChange={(e) => setSupervisor(e.target.value)}
      >
        <option value="" disabled>Select supervisor</option>
        {supervisorOptions.map((opt) => <option key={opt}>{opt}</option>)}
      </select>
      {supervisor === "Other" && (
        <input
          type="text"
          placeholder="Enter supervisor name"
          value={customSupervisor}
          onChange={(e) => setCustomSupervisor(e.target.value)}
          style={styles.input}
        />
      )}

      <label style={styles.label}>Vehicle Number</label>
      <select
        style={styles.select}
        value={vehicleNumber}
        onChange={(e) => setVehicleNumber(e.target.value)}
      >
        <option value="" disabled>Select vehicle number</option>
        {vehicleOptions.map((opt) => <option key={opt}>{opt}</option>)}
      </select>
      {vehicleNumber === "Other" && (
        <input
          type="text"
          placeholder="Enter vehicle number"
          value={customVehicleNumber}
          onChange={(e) => setCustomVehicleNumber(e.target.value)}
          style={styles.input}
        />
      )}

      <label style={styles.label}>Select Model</label>
      <select
        style={styles.select}
        value={selectedModel}
        onChange={(e) => setSelectedModel(e.target.value)}
      >
        <option value="" disabled>Select model</option>
        {modelOptions.map((opt) => <option key={opt}>{opt}</option>)}
      </select>

      <div style={{ display: "flex", gap: "10px", marginTop: 10 }}>
        <button onClick={startProcessing} disabled={processing} style={styles.button}>
          Start
        </button>
        <button onClick={stopProcessing} disabled={!processing} style={styles.stopButton}>
          Stop
        </button>
      </div>

      {result && (
        <div style={{ marginTop: 20 }}>
          <h3>Result</h3>
          <p>Total Count: {result.count}</p>
          <p>Start Time: {result.start_time}</p>
          <p>End Time: {result.end_time}</p>
          <p>Duration (sec): {result.processing_duration_sec}</p>
          <p>Model Used: {result.model_used}</p>
          <p>Classes Detected: {result.classes_detected?.join(", ") || "None"}</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: { maxWidth: 600, margin: "20px auto", padding: 20, fontFamily: "Segoe UI" },
  title: { textAlign: "center", color: "#007acc" },
  input: { width: "100%", padding: 10, marginBottom: 10 },
  label: { fontWeight: "bold", marginBottom: 5, display: "block" },
  select: { width: "100%", padding: 10, marginBottom: 10 },
  button: { padding: 12, backgroundColor: "#28a745", color: "white", border: "none", borderRadius: 6, cursor: "pointer" },
  stopButton: { padding: 12, backgroundColor: "#dc3545", color: "white", border: "none", borderRadius: 6, cursor: "pointer" },
};

export default ProcessingPage;
