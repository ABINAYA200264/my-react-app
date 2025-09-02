

// import React, { useState, useEffect, useRef } from "react";

// function ProcessingPage() {
//   const [videoPath, setVideoPath] = useState("");
//   const [supervisor, setSupervisor] = useState("");
//   const [vehicleNumber, setVehicleNumber] = useState("");
//   const [customSupervisor, setCustomSupervisor] = useState("");
//   const [customVehicleNumber, setCustomVehicleNumber] = useState("");
//   const [selectedModel, setSelectedModel] = useState("");
//   const [processing, setProcessing] = useState(false);
//   const [result, setResult] = useState(null);

//   const pollingIntervalRef = useRef(null);

//   const supervisorOptions = ["Supervisor 1", "Supervisor 2", "Supervisor 3", "Other"];
//   const vehicleOptions = ["Vehicle A123", "Vehicle B456", "Vehicle C789", "Other"];
//   const modelOptions = ["Single Box", "Multiple Box", "4_5_6 Box"];

//   const finalSupervisor = supervisor === "Other" ? customSupervisor : supervisor;
//   const finalVehicleNumber = vehicleNumber === "Other" ? customVehicleNumber : vehicleNumber;

//   const startProcessing = async () => {
//     if (!videoPath) {
//       alert("Please enter the full local video file path");
//       return;
//     }
//     if (!supervisor || !vehicleNumber || !selectedModel) {
//       alert("Please fill all fields before starting processing");
//       return;
//     }

//     setProcessing(true);
//     setResult(null);

//     const formData = new FormData();
//     formData.append("video_url", videoPath);
//     formData.append("supervisor_name", finalSupervisor);
//     formData.append("vehicle_no", finalVehicleNumber);
//     formData.append("selected_model", selectedModel);

//     try {
//       const response = await fetch("http://localhost:8000/process-video", {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok) {
//         const err = await response.json();
//         alert(`Error starting processing: ${err.detail || "Unknown error"}`);
//         setProcessing(false);
//         return;
//       }

//       startPolling();
//     } catch (error) {
//       alert("Failed to start processing: " + error.message);
//       setProcessing(false);
//     }
//   };

//   const startPolling = () => {
//     if (pollingIntervalRef.current) {
//       clearInterval(pollingIntervalRef.current);
//     }
//     pollingIntervalRef.current = setInterval(async () => {
//       try {
//         const res = await fetch("http://localhost:8000/processing-result");
//         const data = await res.json();
//         if (data.result) {
//           setResult(data.result);
//         }
//         if (data.status === "done") {
//           setProcessing(false);
//           clearInterval(pollingIntervalRef.current);
//           pollingIntervalRef.current = null;
//         }
//       } catch (error) {
//         console.error("Error polling processing result:", error);
//       }
//     }, 2000);
//   };

//   useEffect(() => {
//     return () => {
//       if (pollingIntervalRef.current) {
//         clearInterval(pollingIntervalRef.current);
//       }
//     };
//   }, []);

//   const stopProcessing = async () => {
//     try {
//       const response = await fetch("http://localhost:8000/stop-processing", {
//         method: "POST",
//       });
//       if (!response.ok) {
//         return;
//       }
//       setProcessing(false);
//       if (pollingIntervalRef.current) {
//         clearInterval(pollingIntervalRef.current);
//         pollingIntervalRef.current = null;
//       }
//     } catch (error) {
//       // optionally handle error
//     }
//   };

//   return (
//     <div style={styles.container}>
//       <h2 style={styles.title}> 🎥 Video Detection with Bounding Boxes</h2>

//       <input
//         type="text"
//         placeholder="Paste full local video file path here"
//         value={videoPath}
//         onChange={(e) => setVideoPath(e.target.value)}
//         style={styles.input}
//       />

//       <label style={styles.label}>Supervisor</label>
//       <select style={styles.select} value={supervisor} onChange={(e) => setSupervisor(e.target.value)} required>
//         <option value="" disabled>
//           Select supervisor
//         </option>
//         {supervisorOptions.map((opt) => (
//           <option key={opt} value={opt}>
//             {opt}
//           </option>
//         ))}
//       </select>
//       {supervisor === "Other" && (
//         <input
//           type="text"
//           placeholder="Enter supervisor name"
//           value={customSupervisor}
//           onChange={(e) => setCustomSupervisor(e.target.value)}
//           required
//           style={styles.input}
//         />
//       )}

//       <label style={styles.label}>Vehicle Number</label>
//       <select style={styles.select} value={vehicleNumber} onChange={(e) => setVehicleNumber(e.target.value)} required>
//         <option value="" disabled>
//           Select vehicle number
//         </option>
//         {vehicleOptions.map((opt) => (
//           <option key={opt} value={opt}>
//             {opt}
//           </option>
//         ))}
//       </select>
//       {vehicleNumber === "Other" && (
//         <input
//           type="text"
//           placeholder="Enter vehicle number"
//           value={customVehicleNumber}
//           onChange={(e) => setCustomVehicleNumber(e.target.value)}
//           required
//           style={styles.input}
//         />
//       )}

//       <label style={styles.label}>Select Model</label>
//       <select style={styles.select} value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} required>
//         <option value="" disabled>
//           Select model
//         </option>
//         {modelOptions.map((opt) => (
//           <option key={opt} value={opt}>
//             {opt}
//           </option>
//         ))}
//       </select>

//       <button
//         onClick={startProcessing}
//         disabled={processing}
//         style={{ ...styles.button, marginTop: 10, backgroundColor: "#28a745", marginRight: 10 }}
//       >
//         Start
//       </button>
//       <button onClick={stopProcessing} disabled={!processing} style={{ ...styles.button, marginTop: 10, backgroundColor: "#dc3545" }}>
//         Stop
//       </button>

//       {result && 
//           (result.count || result.start_time || result.end_time || (result.classes_detected && result.classes_detected.length > 0))
//       && (
//         <div style={{ marginTop: 20 }}>
//           <h3>Result</h3>
//           <p>Total Count: {result.count}</p>
//           <p>Start Time: {result.start_time}</p>
//           <p>End Time: {result.end_time}</p>
//           <p>Processing Duration (sec): {result.processing_duration_sec}</p>
//           <p>Model Used: {result.model_used}</p>
//           <p>Classes Detected: {result?.classes_detected?.join(", ") || "None"}</p>
//         </div>
//       )}
//     </div>
//   );
// }

// const styles = {
//   container: {
//     maxWidth: 600,
//     margin: "40px auto",
//     padding: 50,
//     fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
//     backgroundColor: "#bed4e5ff",
//     borderRadius: 20,
//     boxShadow: "0 4px 12px rgba(67, 57, 57, 0.1)",
//   },

//   title: {
//     color: "#007acc",
//     textAlign: "center",
//     marginBottom: 20,
//   },
//   input: {
//     width: "100%",
//     padding: "12px 15px",
//     fontSize: 16,
//     marginBottom: 36,
//     borderRadius: 6,
//     border: "1px solid #007acc",
//     outline: "none",
//   },
//   label: {
//     fontWeight: "bold",
//     color: "#007acc",
//     marginBottom: 6,
//     display: "block",
//     textAlign: "left",
//   },
//   select: {
//     width: "100%",
//     padding: "12px 15px",
//     fontSize: 16,
//     marginBottom: 16,
//     borderRadius: 6,
//     border: "1px solid #007acc",
//     outline: "none",
//     backgroundColor: "white",
//   },
//   button: {
//     padding: 14,
//     fontSize: 18,
//     color: "white",
//     border: "none",
//     borderRadius: 10,
//     fontWeight: "bold",
//     cursor: "pointer",
//     transition: "background-color 0.3s ease",
//   },
// };

// export default ProcessingPage;


// import React, { useState, useEffect, useRef } from "react";

// function ProcessingPage() {
//   const [videoPath, setVideoPath] = useState("");
//   const [supervisor, setSupervisor] = useState("");
//   const [vehicleNumber, setVehicleNumber] = useState("");
//   const [customSupervisor, setCustomSupervisor] = useState("");
//   const [customVehicleNumber, setCustomVehicleNumber] = useState("");
//   const [selectedModel, setSelectedModel] = useState("");
//   const [processing, setProcessing] = useState(false);
//   const [result, setResult] = useState(null);

//   const [startHover, setStartHover] = useState(false);
//   const [stopHover, setStopHover] = useState(false);

//   const pollingIntervalRef = useRef(null);

//   const supervisorOptions = ["Supervisor 1", "Supervisor 2", "Supervisor 3", "Other"];
//   const vehicleOptions = ["Vehicle A123", "Vehicle B456", "Vehicle C789", "Other"];
//   const modelOptions = ["Single Box", "Multiple Box", "4_5_6 Box"];

//   const finalSupervisor = supervisor === "Other" ? customSupervisor : supervisor;
//   const finalVehicleNumber = vehicleNumber === "Other" ? customVehicleNumber : vehicleNumber;

//   const startProcessing = async () => {
//     if (!videoPath) {
//       alert("Please enter the full local video file path");
//       return;
//     }
//     if (!supervisor || !vehicleNumber || !selectedModel) {
//       alert("Please fill all fields before starting processing");
//       return;
//     }

//     setProcessing(true);
//     setResult(null);

//     const formData = new FormData();
//     formData.append("video_url", videoPath);
//     formData.append("supervisor_name", finalSupervisor);
//     formData.append("vehicle_no", finalVehicleNumber);
//     formData.append("selected_model", selectedModel);

//     try {
//       const response = await fetch("http://localhost:8000/process-video", {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok) {
//         const err = await response.json();
//         alert(`Error starting processing: ${err.detail || "Unknown error"}`);
//         setProcessing(false);
//         return;
//       }

//       startPolling();
//     } catch (error) {
//       alert("Failed to start processing: " + error.message);
//       setProcessing(false);
//     }
//   };

//   const startPolling = () => {
//     if (pollingIntervalRef.current) {
//       clearInterval(pollingIntervalRef.current);
//     }
//     pollingIntervalRef.current = setInterval(async () => {
//       try {
//         const res = await fetch("http://localhost:8000/processing-result");
//         const data = await res.json();
//         if (data.result) {
//           setResult(data.result);
//         }
//         if (data.status === "done") {
//           setProcessing(false);
//           clearInterval(pollingIntervalRef.current);
//           pollingIntervalRef.current = null;
//         }
//       } catch (error) {
//         console.error("Error polling processing result:", error);
//       }
//     }, 2000);
//   };

//   useEffect(() => {
//     return () => {
//       if (pollingIntervalRef.current) {
//         clearInterval(pollingIntervalRef.current);
//       }
//     };
//   }, []);

//   const stopProcessing = async () => {
//     try {
//       const response = await fetch("http://localhost:8000/stop-processing", {
//         method: "POST",
//       });
//       if (!response.ok) {
//         return;
//       }
//       setProcessing(false);
//       if (pollingIntervalRef.current) {
//         clearInterval(pollingIntervalRef.current);
//         pollingIntervalRef.current = null;
//       }
//     } catch (error) {
//       // optionally handle error
//     }
//   };

//   return (
//     <>
//       <div style={styles.pageBackground}></div>
//       <div style={styles.overlay}></div>
//       <div style={styles.pageWrapper}></div>
//       <div style={styles.container}>
//         <h2 style={styles.title}> 🎥 Video Detection with Bounding Boxes</h2>

//         <input
//           type="text"
//           placeholder="Paste full local video file path here"
//           value={videoPath}
//           onChange={(e) => setVideoPath(e.target.value)}
//           style={styles.input}
//         />

//         <label style={styles.label}>Supervisor</label>
//         <select
//           style={styles.select}
//           value={supervisor}
//           onChange={(e) => setSupervisor(e.target.value)}
//           required
//         >
//           <option value="" disabled>
//             Select supervisor
//           </option>
//           {supervisorOptions.map((opt) => (
//             <option key={opt} value={opt}>
//               {opt}
//             </option>
//           ))}
//         </select>
//         {supervisor === "Other" && (
//           <input
//             type="text"
//             placeholder="Enter supervisor name"
//             value={customSupervisor}
//             onChange={(e) => setCustomSupervisor(e.target.value)}
//             required
//             style={styles.input}
//           />
//         )}

//         <label style={styles.label}>Vehicle Number</label>
//         <select
//           style={styles.select}
//           value={vehicleNumber}
//           onChange={(e) => setVehicleNumber(e.target.value)}
//           required
//         >
//           <option value="" disabled>
//             Select vehicle number
//           </option>
//           {vehicleOptions.map((opt) => (
//             <option key={opt} value={opt}>
//               {opt}
//             </option>
//           ))}
//         </select>
//         {vehicleNumber === "Other" && (
//           <input
//             type="text"
//             placeholder="Enter vehicle number"
//             value={customVehicleNumber}
//             onChange={(e) => setCustomVehicleNumber(e.target.value)}
//             required
//             style={styles.input}
//           />
//         )}

//         <label style={styles.label}>Select Model</label>
//         <select
//           style={styles.select}
//           value={selectedModel}
//           onChange={(e) => setSelectedModel(e.target.value)}
//           required
//         >
//           <option value="" disabled>
//             Select model
//           </option>
//           {modelOptions.map((opt) => (
//             <option key={opt} value={opt}>
//               {opt}
//             </option>
//           ))}
//         </select>
//       <div style={{ display: "flex", gap: "10px", marginTop: 10 }}></div>
//         <button
//           onClick={startProcessing}
//           disabled={processing}
//           style={{
//             ...styles.button,
//             width: 250,  // set Start button width to 120px or any value
//             marginRight: 10,
//             ...(startHover ? styles.buttonHover : {}),
//           }}
//           onMouseEnter={() => setStartHover(true)}
//           onMouseLeave={() => setStartHover(false)}
//         >
//           Start
//         </button>
//         <button
//           onClick={stopProcessing}
//           disabled={!processing}
//           style={{
//             ...styles.button,
//             ...styles.stopButton,
//             width: 250,  // set Stop button width to 120px or any value
//             ...(stopHover ? styles.stopButtonHover : {}),
//           }}
//           onMouseEnter={() => setStopHover(true)}
//           onMouseLeave={() => setStopHover(false)}
//         >
//           Stop
//         </button>

//         {result &&
//           (result.count ||
//             result.start_time ||
//             result.end_time ||
//             (result.classes_detected && result.classes_detected.length > 0)) && (
//             <div style={{ marginTop: 20 }}>
//               <h3>Result</h3>
//               <p>Total Count: {result.count}</p>
//               <p>Start Time: {result.start_time}</p>
//               <p>End Time: {result.end_time}</p>
//               <p>Processing Duration (sec): {result.processing_duration_sec}</p>
//               <p>Model Used: {result.model_used}</p>
//               <p>Classes Detected: {result?.classes_detected?.join(", ") || "None"}</p>
//             </div>
//           )}
//       </div>
//     </>
//   );
// }

// const styles = {
//   pageBackground: {
//     position: "fixed",
//     top: 0,
//     left: 0,
//     right: 0,
//     bottom: 0,
//     backgroundImage: "url('/image.png')",
//     backgroundSize: "contain",
//     backgroundPosition: "center",

//     zIndex: -2,
//   },
  
//   overlay: {
//     position: "fixed",
//     top: 0,
//     left: 0,
//     right: 0,
//     bottom: 0,
//     backgroundColor: "rgba(16, 16, 16, 0.7)",
//     zIndex: -1,
//   },

//    pageWrapper: {
//     display: "flex",
//     justifyContent: "center",
//     alignItems: "center",
//     height: "15vh",
//     width: "15vw",
//   },
//   container: {
//     maxWidth: 600,
//     minHeight: 400,
//     margin: "10px auto",
//     padding: 70,
//     fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
//     backgroundColor: "rgba(114, 175, 190, 0.9)", // white translucent background
//     borderRadius: 30,
//     boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
//     position: "relative",
//     zIndex: 1,
//   },
//   title: {
//     color: "#007acc",
//     textAlign: "center",
//     marginBottom: 20,
//   },
//   input: {
//     width: "100%",
//     padding: "12px 15px",
//     fontSize: 16,
//     marginBottom: 36,
//     borderRadius: 6,
//     border: "1px solid #007acc",
//     outline: "none",
//   },
//   label: {
//     fontWeight: "bold",
//     color: "#007acc",
//     marginBottom: 6,
//     display: "block",
//     textAlign: "left",
//   },
//   select: {
//     width: "100%",
//     padding: "12px 15px",
//     fontSize: 16,
//     marginBottom: 16,
//     borderRadius: 6,
//     border: "1px solid #007acc",
//     outline: "none",
//     backgroundColor: "white",
//   },
//   button: {
//     padding: 14,
//     fontSize: 18,
//     color: "white",
//     border: "none",
//     borderRadius: 10,
//     fontWeight: "bold",
//     cursor: "pointer",
//     transition: "all 0.3s ease",
//     backgroundColor: "#28a745", // default green for start button
//     marginTop: 20,
//     marginRight: 20,
//     boxShadow: "none",
//   },
//   stopButton: {
//     backgroundColor: "#dc3545", // red for stop button
//     marginTop: 20,
//     boxShadow: "none",
//   },
//   buttonHover: {
//     boxShadow: "0 0 20px 3px rgba(40, 167, 69, 0.8)", // green glow
//   },
//   stopButtonHover: {
//     boxShadow: "0 0 20px 3px rgba(220, 53, 69, 0.8)", // red glow
//   },
// };

// export default ProcessingPage;



//--------------------------- above code perfectly work on video url below is for live stream----------------------------------------



// import React, { useState, useEffect, useRef } from "react";
// import ReactHlsPlayer from 'react-hls-player';



// function ProcessingPage() {
//   const [videoPath, setVideoPath] = useState("");
//   const [supervisor, setSupervisor] = useState("");
//   const [vehicleNumber, setVehicleNumber] = useState("");
//   const [customSupervisor, setCustomSupervisor] = useState("");
//   const [customVehicleNumber, setCustomVehicleNumber] = useState("");
//   const [selectedModel, setSelectedModel] = useState("");
//   const [processing, setProcessing] = useState(false);
//   const [result, setResult] = useState(null);
//   const API_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

//   const [startHover, setStartHover] = useState(false);
//   const [stopHover, setStopHover] = useState(false);

//   const pollingIntervalRef = useRef(null);

//   const supervisorOptions = ["Supervisor 1", "Supervisor 2", "Supervisor 3", "Other"];
//   const vehicleOptions = ["Vehicle A123", "Vehicle B456", "Vehicle C789", "Other"];
//   const modelOptions = ["Single Box", "Multiple Box", "4_5_6 Box"];

//   const finalSupervisor = supervisor === "Other" ? customSupervisor : supervisor;
//   const finalVehicleNumber = vehicleNumber === "Other" ? customVehicleNumber : vehicleNumber;

//   const startProcessing = async () => {
//     if (!videoPath) {
//       alert("Please enter the full local video file path");
//       return;
//     }
//     if (!supervisor || !vehicleNumber || !selectedModel) {
//       alert("Please fill all fields before starting processing");
//       return;
//     }

//     setProcessing(true);
//     setResult(null);

//     const formData = new FormData();
//     formData.append("video_url", videoPath);
//     formData.append("supervisor_name", finalSupervisor);
//     formData.append("vehicle_no", finalVehicleNumber);
//     formData.append("selected_model", selectedModel);

//     try {
//       const response = await fetch(`${API_URL}/process-video`, {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok) {
//         const err = await response.json();
//         alert(`Error starting processing: ${err.detail || "Unknown error"}`);
//         setProcessing(false);
//         return;
//       }

//       startPolling();
//     } catch (error) {
//       alert("Failed to start processing: " + error.message);
//       setProcessing(false);
//     }
//   };

//   const startPolling = () => {
//     if (pollingIntervalRef.current) {
//       clearInterval(pollingIntervalRef.current);
//     }
//     pollingIntervalRef.current = setInterval(async () => {
//       try {
//         const res = await fetch("http://localhost:8000/processing-result");
//         const data = await res.json();
//         if (data.result) {
//           setResult(data.result);
//         }
//         if (data.status === "done") {
//           setProcessing(false);
//           clearInterval(pollingIntervalRef.current);
//           pollingIntervalRef.current = null;
//         }
//       } catch (error) {
//         console.error("Error polling processing result:", error);
//       }
//     }, 2000);
//   };

//   useEffect(() => {
//     return () => {
//       if (pollingIntervalRef.current) {
//         clearInterval(pollingIntervalRef.current);
//       }
//     };
//   }, []);

//   const stopProcessing = async () => {
//     try {
//       const response = await fetch("http://localhost:8000/stop-processing", {
//         method: "POST",
//       });
//       if (!response.ok) {
//         return;
//       }
//       setProcessing(false);
//       if (pollingIntervalRef.current) {
//         clearInterval(pollingIntervalRef.current);
//         pollingIntervalRef.current = null;
//       }
//     } catch (error) {
//       // optionally handle error
//     }
//   };

//   return (
//     <>
//       <div style={styles.pageBackground}></div>
//       <div style={styles.overlay}></div>
//       <div style={styles.pageWrapper}></div>
//       <div style={styles.container}>
//         <h2 style={styles.title}> 🎥 Video Detection with Bounding Boxes</h2>

//         <input
//           type="text"
//           placeholder="Paste full local video file path here"
//           value={videoPath}
//           onChange={(e) => setVideoPath(e.target.value)}
//           style={styles.input}
//         />

//         <label style={styles.label}>Supervisor</label>
//         <select
//           style={styles.select}
//           value={supervisor}
//           onChange={(e) => setSupervisor(e.target.value)}
//           required
//         >
//           <option value="" disabled>
//             Select supervisor
//           </option>
//           {supervisorOptions.map((opt) => (
//             <option key={opt} value={opt}>
//               {opt}
//             </option>
//           ))}
//         </select>
//         {supervisor === "Other" && (
//           <input
//             type="text"
//             placeholder="Enter supervisor name"
//             value={customSupervisor}
//             onChange={(e) => setCustomSupervisor(e.target.value)}
//             required
//             style={styles.input}
//           />
//         )}

//         <label style={styles.label}>Vehicle Number</label>
//         <select
//           style={styles.select}
//           value={vehicleNumber}
//           onChange={(e) => setVehicleNumber(e.target.value)}
//           required
//         >
//           <option value="" disabled>
//             Select vehicle number
//           </option>
//           {vehicleOptions.map((opt) => (
//             <option key={opt} value={opt}>
//               {opt}
//             </option>
//           ))}
//         </select>
//         {vehicleNumber === "Other" && (
//           <input
//             type="text"
//             placeholder="Enter vehicle number"
//             value={customVehicleNumber}
//             onChange={(e) => setCustomVehicleNumber(e.target.value)}
//             required
//             style={styles.input}
//           />
//         )}

//         <label style={styles.label}>Select Model</label>
//         <select
//           style={styles.select}
//           value={selectedModel}
//           onChange={(e) => setSelectedModel(e.target.value)}
//           required
//         >
//           <option value="" disabled>
//             Select model
//           </option>
//           {modelOptions.map((opt) => (
//             <option key={opt} value={opt}>
//               {opt}
//             </option>
//           ))}
//         </select>
//       <div style={{ display: "flex", gap: "10px", marginTop: 10 }}></div>
//         <button
//           onClick={startProcessing}
//           disabled={processing}
//           style={{
//             ...styles.button,
//             width: 250,  // set Start button width to 120px or any value
//             marginRight: 10,
//             ...(startHover ? styles.buttonHover : {}),
//           }}
//           onMouseEnter={() => setStartHover(true)}
//           onMouseLeave={() => setStartHover(false)}
//         >
//           Start
//         </button>
//         <button
//           onClick={stopProcessing}
//           disabled={!processing}
//           style={{
//             ...styles.button,
//             ...styles.stopButton,
//             width: 250,  // set Stop button width to 120px or any value
//             ...(stopHover ? styles.stopButtonHover : {}),
//           }}
//           onMouseEnter={() => setStopHover(true)}
//           onMouseLeave={() => setStopHover(false)}
//         >
//           Stop
//         </button>

//         {result && result.hls_url && (
//           result.count ||
//           result.start_time ||
//           result.end_time ||
//           (result.classes_detected && result.classes_detected.length > 0)
//         ) && (
//           <div style={{ marginTop: 20 }}>
//             <h3>Result</h3>
//               <p>Total Count: {result.count}</p>
//               <p>Start Time: {result.start_time}</p>
//               <p>End Time: {result.end_time}</p>
//               <p>Processing Duration (sec): {result.processing_duration_sec}</p>
//               <p>Model Used: {result.model_used}</p>
//               <p>Classes Detected: {result?.classes_detected?.join(", ") || "None"}</p>
//             </div>
//           )}

//         {result && result.hls_url && (
//           <div style={{ marginTop: 20 }}>
//             <h3>Video Stream</h3>
//             <ReactHlsPlayer
//               src={result.hls_url}
//               autoPlay={false}
//               controls={true}
//               width="100%"
//               height="auto"
//             />
//           </div>
//         )}
//       </div>
//     </>
//   );
// }

// const styles = {
//   pageBackground: {
//     position: "fixed",
//     top: 0,
//     left: 0,
//     right: 0,
//     bottom: 0,
//     backgroundImage: "url('/image.png')",
//     backgroundSize: "contain",
//     backgroundPosition: "center",

//     zIndex: -2,
//   },
  
//   overlay: {
//     position: "fixed",
//     top: 0,
//     left: 0,
//     right: 0,
//     bottom: 0,
//     backgroundColor: "rgba(16, 16, 16, 0.7)",
//     zIndex: -1,
//   },


//    pageWrapper: {
//     display: "flex",
//     justifyContent: "center",
//     alignItems: "center",
//     height: "15vh",
//     width: "15vw",
//   },
//   container: {
//     maxWidth: 600,
//     minHeight: 400,
//     margin: "10px auto",
//     padding: 70,
//     fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
//     backgroundColor: "rgba(114, 175, 190, 0.9)", // white translucent background
//     borderRadius: 30,
//     boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
//     position: "relative",
//     zIndex: 1,
//   },
//   title: {
//     color: "#007acc",
//     textAlign: "center",
//     marginBottom: 20,
//   },
//   input: {
//     width: "100%",
//     padding: "12px 15px",
//     fontSize: 16,
//     marginBottom: 36,
//     borderRadius: 6,
//     border: "1px solid #007acc",
//     outline: "none",
//   },
//   label: {
//     fontWeight: "bold",
//     color: "#007acc",
//     marginBottom: 6,
//     display: "block",
//     textAlign: "left",
//   },
//   select: {
//     width: "100%",
//     padding: "12px 15px",
//     fontSize: 16,
//     marginBottom: 16,
//     borderRadius: 6,
//     border: "1px solid #007acc",
//     outline: "none",
//     backgroundColor: "white",
//   },
//   button: {
//     padding: 14,
//     fontSize: 18,
//     color: "white",
//     border: "none",
//     borderRadius: 10,
//     fontWeight: "bold",
//     cursor: "pointer",
//     transition: "all 0.3s ease",
//     backgroundColor: "#28a745", // default green for start button
//     marginTop: 20,
//     marginRight: 20,
//     boxShadow: "none",
//   },
//   stopButton: {
//     backgroundColor: "#dc3545", // red for stop button
//     marginTop: 20,
//     boxShadow: "none",
//   },
//   buttonHover: {
//     boxShadow: "0 0 20px 3px rgba(40, 167, 69, 0.8)", // green glow
//   },
//   stopButtonHover: {
//     boxShadow: "0 0 20px 3px rgba(220, 53, 69, 0.8)", // red glow
//   },
// };

// export default ProcessingPage;

//// above code work perfect on vs code below is render updated one /////

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
  const API_URL = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000";

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
      alert("Please enter the full local video file path");
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
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }
    pollingIntervalRef.current = setInterval(async () => {
      try {
        const res = await fetch("http://localhost:8000/processing-result");
        const data = await res.json();
        if (data.result) {
          setResult(data.result);
        }
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
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const stopProcessing = async () => {
    try {
      const response = await fetch("http://localhost:8000/stop-processing", {
        method: "POST",
      });
      if (!response.ok) {
        return;
      }
      setProcessing(false);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    } catch (error) {
      // optionally handle error
    }
  };

  return (
    <>
      <div style={styles.pageBackground}></div>
      <div style={styles.overlay}></div>
      <div style={styles.pageWrapper}></div>
      <div style={styles.container}>
        <h2 style={styles.title}> 🎥 Video Detection with Bounding Boxes</h2>

        <input
          type="text"
          placeholder="Paste full local video file path here"
          value={videoPath}
          onChange={(e) => setVideoPath(e.target.value)}
          style={styles.input}
        />

        <label style={styles.label}>Supervisor</label>
        <select
          style={styles.select}
          value={supervisor}
          onChange={(e) => setSupervisor(e.target.value)}
          required
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
          style={styles.select}
          value={vehicleNumber}
          onChange={(e) => setVehicleNumber(e.target.value)}
          required
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

        <label style={styles.label}>Select Model</label>
        <select
          style={styles.select}
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          required
        >
          <option value="" disabled>
            Select model
          </option>
          {modelOptions.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
        <div style={{ display: "flex", gap: "10px", marginTop: 10 }}></div>
        <button
          onClick={startProcessing}
          disabled={processing}
          style={{
            ...styles.button,
            width: 250,
            marginRight: 10,
            ...(startHover ? styles.buttonHover : {}),
          }}
          onMouseEnter={() => setStartHover(true)}
          onMouseLeave={() => setStartHover(false)}
        >
          Start
        </button>
        <button
          onClick={stopProcessing}
          disabled={!processing}
          style={{
            ...styles.button,
            ...styles.stopButton,
            width: 250,
            ...(stopHover ? styles.stopButtonHover : {}),
          }}
          onMouseEnter={() => setStopHover(true)}
          onMouseLeave={() => setStopHover(false)}
        >
          Stop
        </button>

        {result && (
          (result.count ||
            result.start_time ||
            result.end_time ||
            (result.classes_detected && result.classes_detected.length > 0)) && (
            <div style={{ marginTop: 20 }}>
              <h3>Result</h3>
              <p>Total Count: {result.count}</p>
              <p>Start Time: {result.start_time}</p>
              <p>End Time: {result.end_time}</p>
              <p>Processing Duration (sec): {result.processing_duration_sec}</p>
              <p>Model Used: {result.model_used}</p>
              <p>Classes Detected: {result?.classes_detected?.join(", ") || "None"}</p>
            </div>
          )
        )}

        {result && result.hls_url && (
          <div style={{ marginTop: 20 }}>
            <h3>Video Stream</h3>
            <ReactHlsPlayer
              src={result.hls_url}
              autoPlay={false}
              controls={true}
              width="100%"
              height="auto"
            />
          </div>
        )}
      </div>
    </>
  );
}

const styles = {
  pageBackground: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundImage: "url('/image.png')",
    backgroundSize: "contain",
    backgroundPosition: "center",
    zIndex: -2,
  },

  overlay: {
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(16, 16, 16, 0.7)",
    zIndex: -1,
  },

  pageWrapper: {
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
    height: "15vh",
    width: "15vw",
  },
  container: {
    maxWidth: 600,
    minHeight: 400,
    margin: "10px auto",
    padding: 70,
    fontFamily: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif",
    backgroundColor: "rgba(114, 175, 190, 0.9)",
    borderRadius: 30,
    boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
    position: "relative",
    zIndex: 1,
  },
  title: {
    color: "#007acc",
    textAlign: "center",
    marginBottom: 20,
  },
  input: {
    width: "100%",
    padding: "12px 15px",
    fontSize: 16,
    marginBottom: 36,
    borderRadius: 6,
    border: "1px solid #007acc",
    outline: "none",
  },
  label: {
    fontWeight: "bold",
    color: "#007acc",
    marginBottom: 6,
    display: "block",
    textAlign: "left",
  },
  select: {
    width: "100%",
    padding: "12px 15px",
    fontSize: 16,
    marginBottom: 16,
    borderRadius: 6,
    border: "1px solid #007acc",
    outline: "none",
    backgroundColor: "white",
  },
  button: {
    padding: 14,
    fontSize: 18,
    color: "white",
    border: "none",
    borderRadius: 10,
    fontWeight: "bold",
    cursor: "pointer",
    transition: "all 0.3s ease",
    backgroundColor: "#28a745",
    marginTop: 20,
    marginRight: 20,
    boxShadow: "none",
  },
  stopButton: {
    backgroundColor: "#dc3545",
    marginTop: 20,
    boxShadow: "none",
  },
  buttonHover: {
    boxShadow: "0 0 20px 3px rgba(40, 167, 69, 0.8)",
  },
  stopButtonHover: {
    boxShadow: "0 0 20px 3px rgba(220, 53, 69, 0.8)",
  },
};

export default ProcessingPage;
