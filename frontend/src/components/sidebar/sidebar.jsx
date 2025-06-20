import React from "react";
import { Link, useLocation } from "react-router-dom";
import "./sidebar.css";

const Sidebar = () => {
  const location = useLocation();

  return (
    <nav className="sidebar">
      <ul>
        <li className={location.pathname === "/" ? "active" : ""}>
          <Link to="/"><i className="fas fa-chart-bar"></i> Dashboard</Link>
        </li>
        <li className={location.pathname === "/identify_by_prompt" ? "active" : ""}>
          <Link to="/identify_by_prompt"><i className="fas fa-search"></i> Identify by Prompt</Link>
        </li>
        <li className={location.pathname === "/identify_by_text" ? "active" : ""}>
          <Link to="/identify_by_text"><i className="fas fa-comments"></i> Identify by Text</Link>
        </li>
      </ul>
    </nav>
  );
};

export default Sidebar;