import React from "react";
import "./Sidebar.css";

const Sidebar = ({ currentPage, onNavigate }) => (
  <nav className="sidebar">
    <ul>
      <li
        className={currentPage === "Dashboard" ? "active" : ""}
        onClick={() => onNavigate("Dashboard")}
      >
        <i className="fas fa-chart-bar"></i> Dashboard
      </li>
      <li
        className={currentPage === "IdentifyByPrompt" ? "active" : ""}
        onClick={() => onNavigate("IdentifyByPrompt")}
      >
        <i className="fas fa-search"></i> Identify By Prompt
      </li>
      <li
        className={currentPage === "IdentifyByText" ? "active" : ""}
        onClick={() => onNavigate("IdentifyByText")}
      >
        <i className="fas fa-comments"></i> Identify By Text
      </li>
    </ul>
  </nav>
);

export default Sidebar;