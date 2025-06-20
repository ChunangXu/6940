import React from "react";
import "./Topbar.css"; // 推荐单独拆出样式

const Topbar = () => (
  <div className="topbar">
    <div className="topbar-title">AI Model Identifier</div>
    <img src="/northeastern-logo.png" alt="Northeastern University" className="topbar-logo" />
  </div>
);

export default Topbar;