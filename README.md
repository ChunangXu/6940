# AI Identity Classifier Platform

This project is a full-stack web platform designed to identify which large language model (LLM) generated a given text or prompt. It combines an intuitive React frontend, a Python-based backend API, and a custom model classifier engine to analyze text features and deliver model predictions with visual explanations.

---

## Project Goals

- Allow users to input a prompt or a full text.
- Analyze the text to determine which AI model (e.g., GPT-4, Claude, Mistral) likely generated it.
- Provide model classification results and evaluation scores.
- Display results visually using PCA plots and keyword frequency maps.

---

## System Architecture

```
[React Frontend] ⇄ [Python API Backend] ⇄ [Classifier Engine + Evaluation Modules + Visualizer]
```

### Core Modules:

| Layer       | Technologies         | Description |
|-------------|----------------------|-------------|
| Frontend    | React.js             | User interface with pages for text/prompt input and dashboard visualization. |
| Backend API | Python (Flask/FastAPI) | Accepts user input, processes requests, and returns model predictions. |
| Classifier  | Python scripts       | Performs model identification based on linguistic and structural features. |
| Evaluation  | MMLU, BoolQ, Custom datasets | Measures performance of different model outputs. |
| Visualizer  | Matplotlib, Seaborn  | Generates PCA plots and keyword analysis charts. |

---

## Main Features

- **Identify by Prompt**: Classify the model behind a short prompt.
- **Identify by Text**: Submit full text and detect its model origin.
- **Dashboard**: View classification results and visual trends.
- **Evaluation Tools**: Test model performance across benchmark datasets.
- **Data Visualizations**: Generate model PCA plots and word frequency heatmaps.

---

## File Structure

```
6940/
├── frontend/                 # React web interface
│   ├── src/pages/            # IdentifyByPrompt, IdentifyByText, Dashboard
│   └── src/components/       # Reusable UI components (Topbar, Sidebar)
├── backend/
│   ├── app.py                # Entry point for backend service
│   ├── ai_identities/
│   │   ├── app/              # API logic & database
│   │   ├── response_classifier/  # Core model detection engine
│   │   ├── performance-evals/   # Benchmark evaluation scripts
│   │   └── Visualiser/      # Plot generation scripts and images
└── README.md
```

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/ChunangXu/6940.git
cd 6940
```

### 2.1 Install the Frontend
```bash
cd frontend
npm install
```

### 2.2 Start the Frontend
```bash
npm start
```

### 3.1 Install the Backend
```bash
cd backend
pip install -r requirements.txt
```

### 3.2 Start the Backend
```bash
uvicorn app:app --reload
```

Make sure the backend server and frontend are running simultaneously. The frontend will communicate with the backend over HTTP (e.g., `localhost:5000`).

---

## Deployment

Unpack the 6940-deployable.zip and follow deployment instuction in its README.md

## Example Visualizations

- `top_words_by_model.png`: Highlights top keywords for each model.
- `model_pca.png`: PCA projection showing model response clusters.

These images are generated by the Visualiser module and displayed on the dashboard.

---

## Future Work

- Add deep learning model classifiers (e.g., fine-tuned transformers).
- Improve UI/UX with live model confidence bars.
- Add user session history and result storage with a database.
- Deploy as a cloud-hosted platform (e.g., using AWS or Vercel).

---

## Contributors

This project is developed by students of Northeastern University as part of a capstone software engineering course.

Special thanks to our sponsor and advisor team for support and guidance.

---

## License

This project is for educational use only. All trademarks or model names (GPT, Claude, etc.) belong to their respective owners.
