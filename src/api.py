#!/usr/bin/env python3
"""
FastAPI Production Inference Service for Historical Immigrant Record Linkage.
Exposes REST endpoints for fuzzy entity scoring, health checks (/healthz),
and FinOps/SRE observability metrics.
Author: Alexandre Borges
"""

import os
import time
from typing import Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model_pipeline import extract_features_for_record, train_and_evaluate

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "record_linkage_rf.joblib")

app = FastAPI(
    title="Historical Immigrant Record Linkage API",
    description=(
        "NLP & Random Forest Entity Resolution microservice for scoring historical "
        "passenger and hostel registration records with orthographic/phonetic variations."
    ),
    version="1.0.0",
)

ARTIFACT = None
METRICS_COUNTER = {"requests_total": 0, "predictions_positive": 0}


@app.on_event("startup")
def load_model() -> None:
    global ARTIFACT
    if not os.path.exists(MODEL_PATH):
        train_and_evaluate()
    ARTIFACT = joblib.load(MODEL_PATH)


class ImmigrantCandidateRequest(BaseModel):
    first_name: str = Field(..., example="Giuseppe")
    last_name: str = Field(..., example="Restino")
    age: float = Field(default=28.0, ge=0.0, le=120.0, example=45.0)
    arrival_year: Optional[int] = Field(
        default=None,
        description="Optional year of arrival. If omitted, uses the blind Text+Demographic model.",
        example=1901,
    )
    family_size: int = Field(default=1, ge=1, example=4)
    steamship: str = Field(default="", example="LES ALPES")
    destination: str = Field(default="", example="FRANCA")
    relationship: str = Field(default="CHEFE", example="CHEFE")


class LinkagePredictionResponse(BaseModel):
    probable_match: bool
    match_probability: float
    model_variant_used: str
    computed_features: Dict[str, float]
    latency_ms: float


@app.get("/healthz", tags=["SRE & Observability"])
def healthcheck() -> Dict[str, str]:
    """Kubernetes liveness and readiness probe endpoint."""
    if ARTIFACT is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded")
    return {"status": "healthy", "model_loaded": "true"}


@app.get("/metrics", tags=["SRE & Observability"])
def metrics() -> Dict[str, int]:
    """Lightweight operational telemetry endpoint."""
    return METRICS_COUNTER


@app.post("/v1/score", response_model=LinkagePredictionResponse, tags=["Inference"])
def score_record(payload: ImmigrantCandidateRequest) -> LinkagePredictionResponse:
    """Score a historical immigrant record using Character N-Gram TF-IDF, Phonetics, and Random Forest."""
    start = time.perf_counter()
    if ARTIFACT is None:
        raise HTTPException(status_code=503, detail="Model artifact not initialized")

    feats = extract_features_for_record(
        nome=payload.first_name,
        sobrenome=payload.last_name,
        idade=payload.age,
        ano_chegada=payload.arrival_year,
        tamanho_familia=payload.family_size,
        vapor=payload.steamship,
        destino=payload.destination,
        parentesco=payload.relationship,
        tfidf_vectorizer=ARTIFACT["vectorizer"],
        targets_tfidf=ARTIFACT["targets_tfidf"],
    )

    if payload.arrival_year is not None:
        cols = ARTIFACT["full_cols"]
        model = ARTIFACT["model_full"]
        variant = "random_forest_full_12_features"
    else:
        cols = ARTIFACT["text_context_cols"]
        model = ARTIFACT["model_text_context"]
        variant = "random_forest_blind_text_demographic_10_features"

    x_vec = np.array([[feats[c] for c in cols]])
    active_features = {c: feats[c] for c in cols}
    prob = float(model.predict_proba(x_vec)[0, 1])
    pred = bool(prob >= 0.5)

    METRICS_COUNTER["requests_total"] += 1
    if pred:
        METRICS_COUNTER["predictions_positive"] += 1

    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return LinkagePredictionResponse(
        probable_match=pred,
        match_probability=round(prob, 4),
        model_variant_used=variant,
        computed_features=active_features,
        latency_ms=latency_ms,
    )
