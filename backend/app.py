import asyncio
import numpy as np
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import datetime
from pydantic import BaseModel
from typing import List
import importlib.util
import sys
import os
import spacy
from collections import Counter

# --- Database setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./identifications.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class IdentificationRecord(Base):
    __tablename__ = "identification_records"
    id = Column(Integer, primary_key=True, index=True)
    input_text = Column(String, nullable=False)
    predicted_model = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- FastAPI setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- NLP and feature extraction ---
nlp = spacy.load("en_core_web_sm")

def generate_word_frequency(texts):
    all_words = []
    for text in texts:
        doc = nlp(text.lower())
        all_words.extend([token.text for token in doc if token.is_alpha])
    return Counter(all_words)

# --- Import ai_identities business logic ---
module_path = os.path.join(os.path.dirname(__file__), "ai_identities", "app", "app.py")
spec = importlib.util.spec_from_file_location("ai_identities_app", module_path)
ai_identities_app = importlib.util.module_from_spec(spec)
sys.modules["ai_identities_app"] = ai_identities_app
spec.loader.exec_module(ai_identities_app)

prepare_features = ai_identities_app.prepare_features
predict_with_confidence = ai_identities_app.predict_with_confidence
query_llm = ai_identities_app.query_llm

# --- Core processing ---
async def process_texts_predict_and_get_results(texts):
    word_freq = await asyncio.to_thread(generate_word_frequency, texts)
    features = await asyncio.to_thread(prepare_features, word_freq)
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

# --- Pydantic models ---
class IdentifyTextRequest(BaseModel):
    texts: List[str]

class GenerateSamplesRequest(BaseModel):
    api_key: str
    provider: str
    model: str
    prompt: str
    num_samples: int = 5
    temperature: float = 0.7

class TestConnectionRequest(BaseModel):
    api_key: str
    provider: str
    model: str
    temperature: float = 0.7

# --- FastAPI routes ---
@app.post("/identify-by-text")
async def identify_text(
    data: IdentifyTextRequest,
    db: Session = Depends(get_db)
):
    if not data.texts or not isinstance(data.texts, list):
        raise HTTPException(status_code=400, detail="Invalid input: 'texts' must be a non-empty list.")
    
    predict_result = await process_texts_predict_and_get_results(data.texts)
    if not predict_result:
        raise HTTPException(status_code=500, detail="Failed to process texts and generate results.")
    
    record = IdentificationRecord(
        input_text=data.texts[0],
        predicted_model=predict_result["predicted_model"],
        confidence=predict_result["confidence_value"]
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    predict_result["record_id"] = record.id
    return predict_result

@app.post("/api/identify-by-prompt")
async def identify_by_prompt(data: GenerateSamplesRequest):
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

@app.post("/api/test-connection")
async def test_connection(data: TestConnectionRequest):
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