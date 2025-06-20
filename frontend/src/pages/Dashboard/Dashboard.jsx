import React from "react";
import "./Dashboard.css";  

const Dashboard = () => (
  <section>
    <h1>Dashboard</h1>
    <p>Summary view with system metrics, logs, and quick links.</p>
    <div className="result-box">
      <h3>System Uptime: 99.9%</h3>
      <p>Total Identification Requests: 258</p>
      <p>Last Update: 2025-06-18</p>
    </div>
  </section>
);

export default Dashboard;