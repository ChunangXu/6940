import React, { useState, useRef } from "react";
import "./IdentifyByPrompt.css";

const IdentifyByPrompt = () => {
  const [provider, setProvider] = useState("openAI");
  const [apiKey, setApiKey] = useState("");
  const [modelName, setModelName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [samples, setSamples] = useState(100);
  const [temperature, setTemperature] = useState(0.5);

  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState("0 / 100");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const intervalRef = useRef(null);

  const testConnection = async () => {
    if (!apiKey || !provider || !modelName) {
      alert("Please fill in all required fields (API key, provider, model).");
      return;
    }
    try {
      const res = await fetch("http://127.0.0.1:8000/api/test-connection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: apiKey,
          provider: provider.toLowerCase(),
          model: modelName,
          temperature: temperature,
          prompt: prompt
        })
      });
      if (!res.ok) throw new Error("Connection failed.");
      const data = await res.json();
      alert("✅ Connection successful!\n\n" + data.message);
    } catch (err) {
      alert("❌ Connection failed. Check API key and model name.");
      console.error("Connection error:", err);
    }
  };

  const identifyModel = () => {
    if (!modelName) {
      alert("Please enter a model name.");
      return;
    }
    setProgress(0);
    setProgressText(`0 / 100`);
    setLoading(true);
    setResult(null);

    intervalRef.current = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(intervalRef.current);

          fetch("http://127.0.0.1:8000/api/identify-by-prompt", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              api_key: apiKey,
              provider: provider.toLowerCase(),
              model: modelName,
              temperature: temperature,
              num_samples: samples,
              prompt: prompt
            })
          })
            .then(res => {
              if (!res.ok) throw new Error("Backend API error");
              return res.json();
            })
            .then(data => {
              setResult(data);
              setLoading(false);
              setProgressText("100 / 100");
            })
            .catch(err => {
              alert("❌ Could not connect to backend. Check if FastAPI is running.");
              setResult({
                predicted_model: "---",
                confidence: "---",
                status: "API error."
              });
              setLoading(false);
              setProgressText("Error");
              console.error("API error:", err);
            });

          return 100;
        }
        setProgressText(`${prev + 10} / 100`);
        return prev + 10;
      });
    }, 200);
  };

  const cancelProcess = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      setProgress(0);
      setProgressText("Cancelled");
      setLoading(false);
    }
  };

  return (
    <section>
      <section className="form-section">
        <label htmlFor="provider">Provider:</label>
        <select id="provider" value={provider} onChange={e => setProvider(e.target.value)}>
          <option value="openAI">OpenAI</option>
          <option value="google">Google</option>
          <option value="deepInfra">DeepInfra</option>
          <option value="anthropic">Anthropic</option>
          <option value="mistral">Mistral</option>
        </select>

        <label htmlFor="apiKey">API Key:</label>
        <input type="password" id="apiKey" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Enter your API Key" />

        <label htmlFor="modelName">Model Name:</label>
        <input type="text" id="modelName" value={modelName} onChange={e => setModelName(e.target.value)} placeholder="e.g., gpt-4o, claude-3-opus" />

        <label htmlFor="prompt">Prompt:</label>
        <input type="text" id="prompt" value={prompt} onChange={e => setPrompt(e.target.value)} placeholder="e.g., Write 10 creative slogans about AI" />

        <label htmlFor="samples">Number of Samples:</label>
        <input type="number" id="samples" value={samples} min={10} max={4000} onChange={e => setSamples(Number(e.target.value))} />

        <label htmlFor="temperature">Temperature:</label>
        <input type="number" id="temperature" value={temperature} min={0} max={1} step={0.1} onChange={e => setTemperature(Number(e.target.value))} />

        <div className="button-group">
          <button onClick={testConnection}>Test Connection</button>
          <button onClick={identifyModel}>Identify Model</button>
          <button onClick={cancelProcess}>Cancel</button>
        </div>
      </section>

      <section className="progress-section">
        <progress id="progressBar" value={progress} max={100}></progress>
        <p id="progressText">{progressText}</p>
      </section>

      <section className="models-section">
        <h3>Models in Database</h3>
        <p>The identifier is trained on these models:</p>
        <div className="tag-list">
          <span className="tag">gpt-4o</span>
          <span className="tag">claude-3-opus</span>
          <span className="tag">gemini-1.5-flash</span>
          <span className="tag">deepseek-r1</span>
          <span className="tag">mistral-7b</span>
        </div>
      </section>

      {loading && (
        <div className="loading-spinner">
          <i className="fas fa-spinner fa-spin"></i>
          <p>Processing...</p>
        </div>
      )}
      {result && (
        <div className="result-box">
          <h3>Predicted Model: <span>{result.predicted_model || "---"}</span></h3>
          <p>Confidence: <span>{result.confidence || "---"}</span></p>
          <p>Analysis: <span>{result.status || "No analysis."}</span></p>
        </div>
      )}
    </section>
  );
};

export default IdentifyByPrompt;