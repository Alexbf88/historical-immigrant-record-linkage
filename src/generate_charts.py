#!/usr/bin/env python3
"""
Generate English high-resolution evaluation charts for the GitHub README.
Author: Alexandre Borges
"""

import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.model_pipeline import (
    TARGET_SURNAMES,
    extract_features_for_record,
    normalize_str,
    parse_age,
    parse_year,
)
from sklearn.feature_extraction.text import TfidfVectorizer

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CCCCCC"
plt.rcParams["axes.linewidth"] = 0.8

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)


def generate_english_charts() -> None:
    df = pd.read_csv(os.path.join(DATA_DIR, "resultados_completos.csv"))
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

    rows = []
    for idx, row in df.iterrows():
        rows.append(
            extract_features_for_record(
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
        )
    df_feat = pd.DataFrame(rows)
    df_feat["Sobrenome"] = df["Sobrenome"].fillna("")
    df_feat["Nome"] = df["Nome"].fillna("")
    df_feat["match_provavel"] = df["match_provavel"].astype(bool)

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

    X_full = df_feat[full_cols].values
    X_tc = df_feat[text_context_cols].values
    y = df_feat["match_provavel"].astype(int).values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
    )
    y_prob_full = cross_val_predict(rf, X_full, y, cv=skf, method="predict_proba")[:, 1]
    y_pred_tc = cross_val_predict(rf, X_tc, y, cv=skf, method="predict")
    y_prob_tc = cross_val_predict(rf, X_tc, y, cv=skf, method="predict_proba")[:, 1]

    # Chart 1: Historical Distribution & NLP Similarity
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    ax1 = axes[0]
    anos_match = df_feat[df_feat["match_provavel"]]["ano_chegada"]
    anos_no = df_feat[~df_feat["match_provavel"]]["ano_chegada"]
    bins = np.arange(1880, 1960, 4)
    ax1.hist(
        [anos_match, anos_no],
        bins=bins,
        stacked=True,
        color=["#0078D4", "#A0AEC0"],
        edgecolor="white",
        label=["Positive Class (Target Family Profile)", "Negative Class (Out of Profile)"],
    )
    ax1.axvspan(1895, 1910, color="#FFC107", alpha=0.2, label="Target Window (1895-1910)")
    ax1.set_title("Historical Distribution by Arrival Year", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Arrival Year at São Paulo Immigrant Hostel", fontsize=9.5)
    ax1.set_ylabel("Unique Immigrant Records", fontsize=9.5)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    ax2 = axes[1]
    top_surnames = (
        df_feat.groupby("Sobrenome")
        .agg(
            qtd=("Nome", "count"),
            sim_lev=("levenshtein_sim_sobrenome", "mean"),
            sim_tfidf=("tfidf_char_sim_sobrenome", "mean"),
        )
        .sort_values(by="qtd", ascending=True)
        .tail(10)
    )
    y_pos = np.arange(len(top_surnames))
    ax2.barh(
        y_pos - 0.18,
        top_surnames["sim_lev"],
        height=0.35,
        color="#0078D4",
        label="Levenshtein Similarity",
    )
    ax2.barh(
        y_pos + 0.18,
        top_surnames["sim_tfidf"],
        height=0.35,
        color="#2B88D8",
        alpha=0.75,
        label="Char N-Gram TF-IDF Cosine Similarity",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{idx} (n={row['qtd']})" for idx, row in top_surnames.iterrows()],
        fontsize=8.5,
    )
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel("Text Similarity Score [0.0 to 1.0]", fontsize=9.5)
    ax2.set_title("NLP Text Mining: Top 10 Extracted Surname Variations", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "distribution_and_nlp_similarity.png"), dpi=300)
    plt.close()

    # Chart 2: Confusion Matrix & ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)
    cm = confusion_matrix(y, y_pred_tc)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"},
        xticklabels=["Non-Match (0)", "Probable Match (1)"],
        yticklabels=["Non-Match (0)", "Probable Match (1)"],
    )
    axes[0].set_title(
        "Confusion Matrix (5-Fold CV - Blind Text & Demographic Model)",
        fontsize=10.5,
        fontweight="bold",
    )
    axes[0].set_xlabel("Predicted Label", fontsize=9.5)
    axes[0].set_ylabel("True Label (Ground Truth)", fontsize=9.5)

    fpr_full, tpr_full, _ = roc_curve(y, y_prob_full)
    auc_full = roc_auc_score(y, y_prob_full)
    fpr_tc, tpr_tc, _ = roc_curve(y, y_prob_tc)
    auc_tc = roc_auc_score(y, y_prob_tc)

    axes[1].plot(
        fpr_full,
        tpr_full,
        color="#0078D4",
        lw=2.2,
        label=f"Full Random Forest Model (AUC = {auc_full:.3f})",
    )
    axes[1].plot(
        fpr_tc,
        tpr_tc,
        color="#107C41",
        lw=2.2,
        linestyle="-.",
        label=f"Blind Text + Demographic Model (AUC = {auc_tc:.3f})",
    )
    axes[1].plot(
        [0, 1],
        [0, 1],
        color="#A0AEC0",
        lw=1.2,
        linestyle="--",
        label="Random Baseline (AUC = 0.500)",
    )
    axes[1].set_title("ROC Curve (5-Fold Stratified Cross-Validation)", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=9.5)
    axes[1].set_ylabel("True Positive Rate (Recall)", fontsize=9.5)
    axes[1].legend(loc="lower right", fontsize=8.5)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "model_performance_roc_cm.png"), dpi=300)
    plt.close()

    # Chart 3: Feature Importance
    rf.fit(X_tc, y)
    importances = rf.feature_importances_
    labels_en = [
        "Immigrant Age (Years)",
        "Family Group Size (Same Book/Page)",
        "Char N-Gram TF-IDF Cosine Similarity",
        "Levenshtein Edit Similarity (Surname)",
        "Sequence Matcher Similarity (Surname)",
        "Phonetic Key Similarity (Surname)",
        "Target Given-Name Similarity",
        "French Steamship Route (Marseille/Le Havre)",
        "Has Recorded Destination Municipality",
        "Head of Household / Solo Traveler Status",
    ]
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    bars = ax.barh(
        range(len(indices)),
        importances[indices],
        color="#0078D4",
        edgecolor="#005A9E",
        height=0.65,
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([labels_en[i] for i in indices], fontsize=9)
    ax.set_xlabel("Relative Feature Importance (Gini Impurity Reduction)", fontsize=9.5)
    ax.set_title(
        "Random Forest Feature Importance (Text Analytics & Demographic Context)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 0.004,
            bar.get_y() + bar.get_height() / 2,
            f"{w * 100:.1f}%",
            va="center",
            fontsize=8.5,
            color="#1F2328",
            fontweight="bold",
        )
    ax.set_xlim(0, max(importances) * 1.18)
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    generate_english_charts()
