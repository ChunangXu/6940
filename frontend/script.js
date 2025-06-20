// Page switching
function showPage(pageId) {
  document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
  document.getElementById(pageId).classList.add('active');

  const menuItems = document.querySelectorAll('.sidebar ul li');
  menuItems.forEach(item => item.classList.remove('active'));
  menuItems.forEach(item => {
    if (item.innerText.toLowerCase().includes(pageId.replace("-", " "))) {
      item.classList.add('active');
    }
  });
}

// Test connection


function testConnection() {
  const prompt = document.getElementById('prompt').value.trim();
  const apiKey = document.getElementById('apiKey').value.trim();
  const provider = document.getElementById('provider').value.trim().toLowerCase();
  const model = document.getElementById('modelName').value.trim();
  const temperature = parseFloat(document.getElementById('temperature').value);

  if (!apiKey || !provider || !model) {
    alert("Please fill in all required fields (API key, provider, model).");
    return;
  }

  fetch("http://127.0.0.1:8000/api/test-connection", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
       api_key: apiKey,
       provider: provider,
       model: model,
       temperature: temperature,
      prompt: prompt
    })
  })
    .then(res => {
      if (!res.ok) throw new Error("Connection failed.");
      return res.json();
    })
    .then(data => {
      alert("✅ Connection successful!\n\n" + data.message);
      console.log("Connection test response:", data);
    })
    .catch(err => {
      alert("❌ Connection failed. Check API key and model name.");
      console.error("Connection error:", err);
    });
}

// Progress bar variables
let progress = 0;
let intervalId = null;

// Real model identification (replaces simulated logic)
function identifyModel() {
  const modelInput = document.getElementById('modelName');
  const prompt = document.getElementById('prompt').value.trim();
  const progressBar = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");
  const modelResult = document.getElementById("result-model");
  const confidenceResult = document.getElementById("result-confidence");
  const analysisResult = document.getElementById("result-analysis");

  const inputText = modelInput.value.trim();

  if (!inputText) {
    alert("Please enter a model response or text.");
    return;
  }

  // Reset progress bar and result
  progress = 0;
  progressBar.value = 0;
  progressText.textContent = "0 / 100";
  modelResult.textContent = "---";
  confidenceResult.textContent = "---";
  analysisResult.textContent = "Processing...";

  // Simulate progress bar
  intervalId = setInterval(() => {
    if (progress >= 100) {
      clearInterval(intervalId);

      // After simulated progress, call backend
      fetch("http://127.0.0.1:8000/api/generate-samples", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
            api_key: document.getElementById('apiKey').value.trim(),
            provider: document.getElementById('provider').value.trim().toLowerCase(),
            model: modelInput.value.trim(),
            temperature: parseFloat(document.getElementById('temperature').value),
            num_samples: parseInt(document.getElementById('sampleCount').value),
            prompt: prompt
        })
      })
        .then(res => {
          if (!res.ok) throw new Error("Backend API error");
          return res.json();
        })
        .then(data => {
          modelResult.textContent = data.predicted_model || "unknown";
          confidenceResult.textContent = data.confidence || "--%";
          analysisResult.textContent = data.status || "No analysis.";
          showPage('results');
        })
        .catch(err => {
          alert("❌ Could not connect to backend. Check if FastAPI is running.");
          console.error("API error:", err);
          analysisResult.textContent = "API error.";
        });

      return;
    }

    progress += 10;
    progressBar.value = progress;
    progressText.textContent = `${progress} / 100`;
  }, 200);
}

// Cancel identification process
function cancelProcess() {
  if (intervalId) {
    clearInterval(intervalId);
    document.getElementById("progressText").textContent = "Cancelled";
    document.getElementById("progressBar").value = 0;
  }
}

// Real-time live response analysis (optional use on results.html if needed)
function analyzeResponse() {
  const text = document.getElementById("aiResponse").value.trim();
  const loader = document.getElementById("loading");
  const resultBox = document.getElementById("liveResultBox");

  if (!text) {
    alert("Please paste an AI response.");
    return;
  }

  loader.style.display = "block";
  resultBox.style.display = "none";

  fetch("http://127.0.0.1:8000/identify-text", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ texts: [text] })
  })
    .then(res => {
      if (!res.ok) throw new Error("Failed to reach backend.");
      return res.json();
    })
    .then(data => {
      loader.style.display = "none";
      resultBox.style.display = "block";

      document.getElementById("result-model-live").textContent = data.predicted_model || "unknown";
      document.getElementById("result-confidence-live").textContent = data.confidence || "--%";
      document.getElementById("result-analysis-live").textContent = data.status || "No analysis.";
    })
    .catch(err => {
      loader.style.display = "none";
      alert("❌ Could not connect to backend. Make sure the API is running.");
      console.error("API error:", err);
    });
}