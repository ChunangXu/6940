from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import importlib.util
import sys
import os

# construct absolute path of app.py file in ai-identities directory
module_path = os.path.join(os.path.dirname(__file__), "ai-identities", "app", "app.py")
spec = importlib.util.spec_from_file_location("ai_identities_app", module_path)
ai_identities_app = importlib.util.module_from_spec(spec)
sys.modules["ai_identities_app"] = ai_identities_app
spec.loader.exec_module(ai_identities_app)

# Import necessary functions from the ai_identities_app module
process_responses = ai_identities_app.process_responses
prepare_features = ai_identities_app.prepare_features
classifier = ai_identities_app.classifier

app = FastAPI()

class IdentifyTextRequest(BaseModel):
    texts: List[str]

@app.post("/identify-text")
async def identify_text(data: IdentifyTextRequest):
    # 1. text to word frequency
    word_freq = process_responses(data.texts)
    # 2. word frequency to features
    features = prepare_features(word_freq)
    # 3. features to prediction
    pred = classifier.predict(features)[0]
    proba = classifier.predict_proba(features)[0]
    return {
        "predicted_model": pred,
        "confidence": float(max(proba)),
        "top_predictions": [
            {"model": m, "probability": float(p)}
            for m, p in zip(classifier.classes_, proba)
        ]
    }