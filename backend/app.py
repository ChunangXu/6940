import asyncio
from http.client import HTTPException
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from pydantic import BaseModel
from typing import List

import importlib.util
import sys
import os
import spacy
from collections import Counter



nlp = spacy.load("en_core_web_sm")

def generate_word_frequency(texts):
    """
    Generate word frequency Counter from a list of texts using spaCy.
    Only alphabetic tokens are counted, all lowercased.
    """
    all_words = []
    for text in texts:
        doc = nlp(text.lower())
        all_words.extend([token.text for token in doc if token.is_alpha])
    return Counter(all_words)

async def process_texts_predict_and_get_results(texts):
    """
    Process a list of texts to generate word frequency and predict model.
    Returns a dictionary with the results.
    """
    # 1. text to word frequency
    word_freq = await asyncio.to_thread(generate_word_frequency, texts)
    # 2. word frequency to features
    features = await asyncio.to_thread(prepare_features, word_freq)
    # 3. features to prediction
    predicted_model, confidence, top_predictions, error = await predict_with_confidence(features)

    status_message = "success"
    final_predicted_model = predicted_model
    if predicted_model == "unknown":
        status_message = "success_unrecognized"
        final_predicted_model = "unrecognized_model"
        confidence = 0.0
        top_predictions = []
    elif np.sum(features) == 0:
        status_message = "success_no_overlap"
        confidence = 0.0

    return {
        "unique_words_extracted": len(word_freq),
        "predicted_model": final_predicted_model,
        "confidence": f"{confidence:.2%}",
        "confidence_value": confidence,
        "top_predictions": top_predictions,
        "word_frequencies_top": dict(word_freq.most_common(20)),
        "status": status_message,
    }

# construct absolute path of app.py file in ai-identities directory
module_path = os.path.join(os.path.dirname(__file__), "ai_identities", "app", "app.py")
spec = importlib.util.spec_from_file_location("ai_identities_app", module_path)
ai_identities_app = importlib.util.module_from_spec(spec)
sys.modules["ai_identities_app"] = ai_identities_app
spec.loader.exec_module(ai_identities_app)

# Import necessary functions from the ai_identities_app module
prepare_features = ai_identities_app.prepare_features
classifier = ai_identities_app.classifier
predict_with_confidence = ai_identities_app.predict_with_confidence
query_llm = ai_identities_app.query_llm


class IdentifyTextRequest(BaseModel):
    texts: List[str]

@app.post("/identify-by-text")
async def identify_text(data: IdentifyTextRequest):
    print("🔍 Received text for identification:", data.texts)
    """
    Endpoint to identify text and predict model based on word frequency.
    Expects a list of texts in the request body.
    """
    if not data.texts or not isinstance(data.texts, list):
        raise HTTPException(status_code=400, detail="Invalid input: 'texts' must be a non-empty list.")
    
    predict_result = await process_texts_predict_and_get_results(data.texts)
    if not predict_result:
        raise HTTPException(status_code=500, detail="Failed to process texts and generate results.")
    return predict_result


class GenerateSamplesRequest(BaseModel):
    api_key: str
    provider: str
    model: str
    prompt: str
    num_samples: int = 100
    temperature: float = 0.7

@app.post("/api/identify-by-prompt")
async def generate_samples(data: GenerateSamplesRequest):
    """
    Endpoint to generate samples using a language model.
    Expects API key, provider, model, prompt, number of samples, and temperature in the request body.
    """
    try:
        responses = await query_llm(
            api_key=data.api_key,
            provider=data.provider,
            model=data.model,
            num_samples=data.num_samples,
            temperature=data.temperature,
            prompt=data.prompt
        )
        if not responses:
            raise HTTPException(status_code=500, detail="No responses generated from the LLM.")
        
        predict_result = await process_texts_predict_and_get_results(responses)
        if not predict_result:
            raise HTTPException(status_code=500, detail="Failed to process generated samples.")
        
        return predict_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate samples: {str(e)}")

class TestConnectionRequest(BaseModel):
    api_key: str
    provider: str
    model: str
    temperature: float = 0.7

@app.post("/api/test-connection")
async def test_connection(data: TestConnectionRequest):
    """
    Test LLM connection with 1 sample
    """
    try:
        responses = await query_llm(
            api_key=data.api_key,
            provider=data.provider,
            model=data.model,
            num_samples=1,
            temperature=data.temperature,
            prompt="Hello"
        )
        if responses and isinstance(responses[0], str):
            return {"status": "success", "message": "API connection successful", "response": responses[0]}
        else:
            raise ValueError("No valid response from model.")
    except Exception as e:
        return {"status": "error", "message": f"Connection failed: {str(e)}"}
