#!/usr/bin/env python3
"""
Text Analytics (NLP) & Supervised Classification Pipeline for Historical Record Linkage.
Trains a Random Forest classifier using character n-gram TF-IDF, Levenshtein/Phonetic
similarity, and demographic context, exporting the model artifact and evaluation metrics.
Author: Alexandre Borges
"""

import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_SURNAMES = ["RASTINI", "RESTINI", "RASTELLI", "PASTINI", "BASTINI"]
TARGET_FIRSTNAMES = [
    "AUGUSTO",
    "AUGUSTE",
    "ANGELINA",
    "ANGELINE",
    "JOSE",
    "JOSEPH",
    "GIUSEPPE",
    "ANGELO",
]
FRENCH_ROUTE_VESSELS = {
    "LES ALPES",
    "ORLEANAIS",
    "BRETAGNE",
    "AQUITAINE",
    "POITOU",
    "AMAZONE",
    "ESPAGNE",
    "PROVENCE",
}


def normalize_str(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).strip().upper()


def parse_age(age_val) -> float:
    """Convert historical age strings ( including infant fractions like '9/12') into float years."""
    if pd.isna(age_val):
        return 25.0
    s = str(age_val).strip()
    if "/" in s:
        parts = s.split("/")
        try:
            return round(float(parts[0]) / float(parts[1]), 2)
        except Exception:
            return 0.5
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else 25.0


def parse_year(date_val) -> float:
    if pd.isna(date_val):
        return 1901.0
    m = re.search(r"\b(18\d{2}|19\d{2})\b", str(date_val))
    return float(m.group(1)) if m else 1901.0


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0]."""
    if not s1 or not s2:
        return 0.0
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return 1.0 - (dp[n] / max(m, n))


def phonetic_key(word: str) -> str:
    """Phonetic normalization for Franco-Italian-Portuguese surname transcriptions."""
    w = re.sub(r"[^A-Z]", "", normalize_str(word))
    w = w.replace("PH", "F").replace("TH", "T").replace("CH", "K")
    w = re.sub(r"[AEIOUY]+", "A", w)
    return re.sub(r"(.)\1+", r"\1", w)


def extract_features_for_record(
    nome: str,
    sobrenome: str,
    idade: float,
    ano_chegada: Optional[float],
    tamanho_familia: int,
    vapor: str,
    destino: str,
    parentesco: str,
    tfidf_vectorizer: TfidfVectorizer,
    targets_tfidf,
) -> Dict[str, float]:
    """Compute NLP and demographic features for a single record (used in training and REST inference)."""
    s_norm = normalize_str(sobrenome)
    n_norm = normalize_str(nome)
    target_phonetics = [phonetic_key(t) for t in TARGET_SURNAMES]

    if s_norm:
        s_vec = tfidf_vectorizer.transform([s_norm])
        tfidf_sim = float(cosine_similarity(s_vec, targets_tfidf).max())
        lev_sim = max(levenshtein_similarity(s_norm, t) for t in TARGET_SURNAMES)
        seq_sim = max(SequenceMatcher(None, s_norm, t).ratio() for t in TARGET_SURNAMES)
        pk = phonetic_key(s_norm)
        phon_sim = max(levenshtein_similarity(pk, tp) for tp in target_phonetics)
    else:
        tfidf_sim = lev_sim = seq_sim = phon_sim = 0.0

    if n_norm:
        tokens = n_norm.split()
        fname_sim = max(
            levenshtein_similarity(tok, tn)
            for tok in tokens
            for tn in TARGET_FIRSTNAMES
        )
    else:
        fname_sim = 0.0

    year_val = float(ano_chegada) if ano_chegada is not None else 1901.0
    dist_window = (
        0.0
        if 1895 <= year_val <= 1910
        else min(abs(year_val - 1895), abs(year_val - 1910))
    )

    return {
        "ano_chegada": year_val,
        "distancia_janela_temporal": dist_window,
        "idade_num": float(idade),
        "tamanho_familia": int(tamanho_familia),
        "tfidf_char_sim_sobrenome": round(tfidf_sim, 4),
        "levenshtein_sim_sobrenome": round(lev_sim, 4),
        "seq_sim_sobrenome": round(seq_sim, 4),
        "fonetica_sim_sobrenome": round(phon_sim, 4),
        "similaridade_prenome_alvo": round(fname_sim, 4),
        "vapor_rota_francesa": 1 if normalize_str(vapor) in FRENCH_ROUTE_VESSELS else 0,
        "tem_destino_registrado": 1 if len(normalize_str(destino)) > 0 else 0,
        "eh_chefe_ou_so": 1
        if normalize_str(parentesco) in {"CHEFE", "CF", "MARIDO", "SO", "SOZINHO"}
        else 0,
    }


def train_and_evaluate() -> None:
    csv_path = os.path.join(DATA_DIR, "resultados_completos.csv")
    df = pd.read_csv(csv_path)

    sobrenomes_norm = df["Sobrenome"].fillna("").apply(normalize_str).tolist()
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    vectorizer.fit(sobrenomes_norm + TARGET_SURNAMES)
    targets_tfidf = vectorizer.transform(TARGET_SURNAMES)

    fam_counts = (
        df.groupby(["Livro", "Página", "Família"])["Nome"]
        .transform("count")
        .fillna(1)
        .astype(int)
    )

    feature_rows: List[Dict[str, float]] = []
    for idx, row in df.iterrows():
        feats = extract_features_for_record(
            nome=str(row.get("Nome", "")),
            sobrenome=str(row.get("Sobrenome", "")),
            idade=parse_age(row.get("Idade")),
            ano_chegada=parse_year(row.get("Data de Chegada")),
            tamanho_familia=int(fam_counts.iloc[idx]),
            vapor=str(row.get("Vapor/Navio", "")),
            destino=str(row.get("Destino", "")),
            parentesco=str(row.get("Parentesco", "")),
            tfidf_vectorizer=vectorizer,
            targets_tfidf=targets_tfidf,
        )
        feature_rows.append(feats)

    df_feats = pd.DataFrame(feature_rows)
    text_context_cols = [
        "idade_num",
        "tamanho_familia",
        "tfidf_char_sim_sobrenome",
        "levenshtein_sim_sobrenome",
        "seq_sim_sobrenome",
        "fonetica_sim_sobrenome",
        "similaridade_prenome_alvo",
        "vapor_rota_francesa",
        "tem_destino_registrado",
        "eh_chefe_ou_so",
    ]
    full_cols = ["ano_chegada", "distancia_janela_temporal"] + text_context_cols

    X_full = df_feats[full_cols].values
    X_text_ctx = df_feats[text_context_cols].values
    y = df["match_provavel"].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf_full = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )
    y_pred_full = cross_val_predict(rf_full, X_full, y, cv=skf, method="predict")
    y_prob_full = cross_val_predict(rf_full, X_full, y, cv=skf, method="predict_proba")[
        :, 1
    ]
    rf_full.fit(X_full, y)

    rf_text_ctx = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )
    y_pred_tc = cross_val_predict(rf_text_ctx, X_text_ctx, y, cv=skf, method="predict")
    y_prob_tc = cross_val_predict(
        rf_text_ctx, X_text_ctx, y, cv=skf, method="predict_proba"
    )[:, 1]
    rf_text_ctx.fit(X_text_ctx, y)

    artifact = {
        "vectorizer": vectorizer,
        "targets_tfidf": targets_tfidf,
        "model_full": rf_full,
        "model_text_context": rf_text_ctx,
        "full_cols": full_cols,
        "text_context_cols": text_context_cols,
    }
    joblib.dump(artifact, os.path.join(MODEL_DIR, "record_linkage_rf.joblib"))

    metrics = {
        "dataset_size": int(len(df)),
        "positive_class_count": int(y.sum()),
        "negative_class_count": int((1 - y).sum()),
        "full_model_cv5": {
            "accuracy": round(float(accuracy_score(y, y_pred_full)), 4),
            "precision": round(float(precision_score(y, y_pred_full)), 4),
            "recall": round(float(recall_score(y, y_pred_full)), 4),
            "f1_score": round(float(f1_score(y, y_pred_full)), 4),
            "roc_auc": round(float(roc_auc_score(y, y_prob_full)), 4),
        },
        "blind_text_context_model_cv5": {
            "accuracy": round(float(accuracy_score(y, y_pred_tc)), 4),
            "precision": round(float(precision_score(y, y_pred_tc)), 4),
            "recall": round(float(recall_score(y, y_pred_tc)), 4),
            "f1_score": round(float(f1_score(y, y_pred_tc)), 4),
            "roc_auc": round(float(roc_auc_score(y, y_prob_tc)), 4),
        },
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Model artifact and metrics saved successfully:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_evaluate()
