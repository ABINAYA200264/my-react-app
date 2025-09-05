import React, { useState } from "react";

const API_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [videoPath, setVideoPath] = useState("");
  const [supervisorName, setSupervisorName] = useState("");
  const [vehicleNo, setVehicleNo] = useState("");
  const [selectedModel, setSelectedModel] = useState("Single Box");
  const [result, setResult] = useState(null);

  const startProcessing = async () => {
    if (!videoPath || !supervisorName || !vehicleNo) {
      alert("Fill all fields");
      return;
    }

    const formData = new FormData();
    formData.append("video_url", videoPath);
    formData.append("supervisor_name", supervisorName);
    formData.append("vehicle_no", vehicleNo);
    formData.append("selected_model", selectedModel);

    const res = await fetch(`${API_URL}/process-video`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    console.log(data);

    // Poll for results
    const interval = setInterval(async () => {
      const res = await fetch(`${API_URL}/results`);
      const json = await res.json();
      if (json.status === "done") {
        setResult(json.result);
        clearInterval(interval);
      }
    }, 5000);
  };

  return (
    <div style={{ padding: "20px", fontFamily: "Arial" }}>
      <h2>📦 Box Detection App</h2>
      <div>
        <input
          type="text"
          placeholder="Video URL or Path"
          value={videoPath}
          onChange={(e) => setVideoPath(e.target.value)}
        />
      </div>
      <div>
        <input
          type="text"
          placeholder="Supervisor Name"
          value={supervisorName}
          onChange={(e) => setSupervisorName(e.target.value)}
        />
      </div>
      <div>
        <input
          type="text"
          placeholder="Vehicle No"
          value={vehicleNo}
          onChange={(e) => setVehicleNo(e.target.value)}
        />
      </div>
      <div>
        <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
          <option>Single Box</option>
          <option>Multiple Box</option>
          <option>4_5_6 Box</option>
        </select>
      </div>
      <button onClick={startProcessing}>🚀 Start Processing</button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <h3>✅ Results</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default App;
