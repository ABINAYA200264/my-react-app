import React, { useState } from "react";

function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [supervisor, setSupervisor] = useState("");
  const [vehicle, setVehicle] = useState("");
  const [status, setStatus] = useState("");
  const [results, setResults] = useState(null);

  const startProcessing = async () => {
    setStatus("Starting...");
    const formData = new FormData();
    formData.append("video_url", videoUrl);
    formData.append("supervisor_name", supervisor);
    formData.append("vehicle_no", vehicle);

    const res = await fetch("http://127.0.0.1:8000/process-video", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setStatus(data.message);
  };

  const checkResults = async () => {
    const res = await fetch("http://127.0.0.1:8000/results");
    const data = await res.json();
    if (data.status === "done") {
      setResults(data.result);
    } else {
      setStatus("No results yet...");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>4_5_6 Box Detection</h2>
      <input
        type="text"
        placeholder="Video URL (file path or stream)"
        value={videoUrl}
        onChange={(e) => setVideoUrl(e.target.value)}
        style={{ width: "100%", marginBottom: "10px" }}
      />
      <input
        type="text"
        placeholder="Supervisor Name"
        value={supervisor}
        onChange={(e) => setSupervisor(e.target.value)}
        style={{ width: "100%", marginBottom: "10px" }}
      />
      <input
        type="text"
        placeholder="Vehicle No"
        value={vehicle}
        onChange={(e) => setVehicle(e.target.value)}
        style={{ width: "100%", marginBottom: "10px" }}
      />
      <button onClick={startProcessing} style={{ marginRight: "10px" }}>
        Start Processing
      </button>
      <button onClick={checkResults}>Check Results</button>
      <p>Status: {status}</p>
      {results && (
        <pre style={{ background: "#eee", padding: "10px" }}>
          {JSON.stringify(results, null, 2)}
        </pre>
      )}
    </div>
  );
}

export default App;
