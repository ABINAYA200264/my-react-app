// import React, { useState } from "react";

// function Login({ onLoginSuccess }) {
//   const [username, setUsername] = useState("");
//   const [password, setPassword] = useState("");
//   const [error, setError] = useState("");

//   const submitLogin = (e) => {
//     e.preventDefault();
//     if (username === "user" && password === "pass") {
//       setError("");
//       onLoginSuccess();
//     } else {
//       setError("Invalid username or password");
//     }
//   };

//   return (
//     <div style={{ maxWidth: 300, margin: "auto", paddingTop: 50 }}>
//       <h2>Login</h2>
//       <form onSubmit={submitLogin}>
//         <input
//           type="text"
//           placeholder="Username"
//           value={username}
//           onChange={(e) => setUsername(e.target.value)}
//           style={{ width: "100%", marginBottom: 10, padding: 8 }}
//         />
//         <input
//           type="password"
//           placeholder="Password"
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//           style={{ width: "100%", marginBottom: 10, padding: 8 }}
//         />
//         <button type="submit" style={{ width: "100%", padding: 8 }}>
//           Login
//         </button>
//       </form>
//       {error && <p style={{ color: "red" }}>{error}</p>}
//     </div>
//   );
// }

// export default Login;

// ----- above code is not reconize the mail id user name not accepted ----- #


import React, { useState } from "react";

function Login({ onLoginSuccess }) {
  const submitLogin = (e) => {
    e.preventDefault();
    // Directly call onLoginSuccess regardless of input
    onLoginSuccess();
  };

  return (
    <form onSubmit={submitLogin} style={{ maxWidth: 300, margin: "auto", paddingTop: 50 }}>
      <h2>Login</h2>
      <input type="email" placeholder="Email" required style={{ width: "100%", marginBottom: 10, padding: 8 }} />
      <input type="password" placeholder="Password" required style={{ width: "100%", marginBottom: 10, padding: 8 }} />
      <button type="submit" style={{ width: "100%", padding: 8 }}>
        Login
      </button>
    </form>
  );
}

export default Login;
