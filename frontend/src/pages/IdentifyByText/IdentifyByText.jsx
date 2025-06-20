import React, { useState } from "react";
import "./IdentifyByText.css";

const IdentifyByText = () => {
  const [aiResponse, setAiResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const analyzeResponse = async () => {
    if (!aiResponse.trim()) {
      alert("Please paste an AI response.");
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/identify-by-text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: [aiResponse] }),
      });
      if (!res.ok) throw new Error("Failed to reach backend.");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      alert("❌ Could not connect to backend. Make sure the API is running.");
      console.error("API error:", err);
    }
    setLoading(false);
  };

  return (
    <section>
      <h1>Identify By Text</h1>
      <p>Display predicted model, confidence, and response analysis.</p>
      <div className="user-input-area">
        <label htmlFor="aiResponse">Paste AI Response Below:</label>
        <textarea
          id="aiResponse"
          value={aiResponse}
          onChange={e => setAiResponse(e.target.value)}
          placeholder="Paste the AI's response here..."
        />
        <div className="button-group">
          <button onClick={analyzeResponse} disabled={loading}>
            {loading ? "Identifying..." : "Identify"}
          </button>
        </div>
      </div>
      {loading && (
        <div className="loading-spinner">
          <i className="fas fa-spinner fa-spin"></i>
          <p>Analyzing...</p>
        </div>
      )}
      {result && (
        <div className="result-box">
          <h3>Predicted Model: <span>{result.predicted_model || "unknown"}</span></h3>
          <p>Confidence: <span>{result.confidence || "--%"}</span></p>
          <p>Analysis: <span>{result.status || "No analysis."}</span></p>
        </div>
      )}
    </section>
  );
};

export default IdentifyByText;