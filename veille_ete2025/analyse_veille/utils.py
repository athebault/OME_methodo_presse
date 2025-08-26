#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import List, Tuple, Dict

from distinctipy import distinctipy
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from wordcloud import WordCloud

    
##----------------------------------------------- Analyse des commentaires --------------------------------------------------##
# Fonction de gestion du texte
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)  # ponctuation
    s = re.sub(r"\d+", " ", s)  # nombres
    s = re.sub(r"\s+", " ", s).strip()
    return s


def remove_stopwords(tokens: List[str], stopwords: List[str]) -> List[str]:
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def tokenize(s: str) -> List[str]:
    return s.split()

def clean_text(s: str, stopwords: List[str]) -> str:
    s = normalize_text(s)
    tokens = remove_stopwords(tokenize(s), stopwords)
    return " ".join(tokens)


def lexicon_sentiment(text: str, positive_words: List[str], negative_words: List[str]) -> Tuple[str, float]:
    tokens = set(tokenize(normalize_text(text)))
    pos = len(tokens & positive_words)
    neg = len(tokens & negative_words)
    score = (pos - neg) / max(1, pos + neg)
    label = "POSITIVE" if score > 0 else ("NEGATIVE" if score < 0 else "NEUTRAL")
    return label, float(score)


def choose_k_by_silhouette(X, k_min=2, k_max=5) -> int:
    if X.shape[0] < k_min + 1:
        return 1  # not enough samples
    best_k, best_score = None, -1.0
    for k in range(k_min, min(k_max, X.shape[0]) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        # Silhouette needs > 1 cluster and < n_samples
        if len(set(labels)) <= 1 or len(set(labels)) >= X.shape[0]:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k or 1

def extract_top_terms_per_cluster(tfidf: TfidfVectorizer, X, labels, topn=10) -> Dict[int, List[str]]:
    terms = np.array(tfidf.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for cl in sorted(set(labels)):
        idx = np.where(labels == cl)[0]
        if len(idx) == 0:
            out[cl] = []
            continue
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        top_idx = centroid.argsort()[::-1][:topn]
        out[cl] = terms[top_idx].tolist()
    return out


## --------------------------------------------- Vizualisations -------------------------------------------------- ##
# Visualisation resultats de veille
def plot_countplots(df, cols, titles, hue="Titre", na_label="NA", max_x=10, n_colors = 21):
    """
    Crée une grille de countplots horizontaux avec :
    - séparation des valeurs par virgule
    - 'NA' en bas
    - une seule légende globale en bas
    """
    n = len(cols)
    rows = (n + 2) // 3  # 3 colonnes par ligne
    fig, ax = plt.subplots(rows, 3, figsize=(20, 4 * rows))
    ax = ax.flatten()

    for i, (col, title) in enumerate(zip(cols, titles)):
        data = df[[col, hue]].fillna(na_label)
        data[col] = data[col].astype(str).str.split(",")
        data = data.explode(col)
        data[col] = data[col].str.strip()

        order = [x for x in data[col].value_counts().index if x != na_label] + [na_label]

        sns.countplot(
            data=data,
            y=col,
            hue=hue,
            order=order,
            palette=distinctipy.get_colors(n_colors),
            ax=ax[i],
            legend=(i == 0)  # légende seulement pour le 1er
        )
        # ligne verticale à x = 5
        ax[i].axvline(x=5, linestyle="--", linewidth=1, alpha=0.9, color="black")

        # Ajuster les labels
        ax[i].set_title(title)
        ax[i].set_xlabel("")
        ax[i].set_ylabel("")
        ax[i].set_xlim(0, max_x)

    # Supprimer axes vides
    for j in range(i + 1, len(ax)):
        fig.delaxes(ax[j])


    # Récupérer handles/labels et placer légende en bas
    # --- Création des labels légende avec nombre de lectures ---
    counts = df[hue].value_counts()
    handles, labels = ax[0].get_legend_handles_labels()

    labels_with_counts = [
        f"{label} (n={counts.get(label, 0)})" for label in labels
    ]
    
    ax[0].legend_.remove()
    
    # --- Création de la légende globale ---
    fig.legend(
        handles, labels_with_counts,
        title=hue,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.65)  # à droite et centré verticalement
    )

    plt.tight_layout()
    plt.show()
    

def plot_kmeans(X, km_model, use_pca=True, title=None):
    """
    X : ndarray de forme (n_samples, n_features)

    use_pca : projeter en 2D si n_features > 2
    """
    labels = km_model.labels_
    centers = km_model.cluster_centers_

    # Projection 2D (PCA si besoin)
    if X.shape[1] > 2 and use_pca:
        pca = PCA(n_components=2, random_state=42)
        X2 = pca.fit_transform(X)
        C2 = pca.transform(centers)
        xlabel, ylabel = "PC1", "PC2"
    else:
        X2 = X[:, :2]
        C2 = centers[:, :2] if centers.shape[1] >= 2 else np.c_[centers, np.zeros((centers.shape[0], 2 - centers.shape[1]))]
        xlabel, ylabel = "x1", "x2"

    # Scatter des points + centres (X noir)
    plt.figure(figsize=(6,5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=25, alpha=0.85)
    plt.scatter(C2[:, 0], C2[:, 1], c="black", s=120, marker="X", label="centres")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title or f"KMeans (k={k})")
    plt.legend()
    plt.tight_layout()
    plt.show()



def make_df(alpha_dict: dict, question: str) -> pd.DataFrame:
    per = alpha_dict[question]["per_label"]
    rows = [{"Label": k, "Alpha": float(v)} for k, v in per.items()]
    df = pd.DataFrame(rows)
    df = df.sort_values("Alpha", ascending=True)
    return df

def make_multi_figure(alpha_dict: dict, titre: str, max_height=220, width=850) -> go.Figure:
    questions = list(alpha_dict.keys())
    dfs = {q: make_df(alpha_dict, q) for q in questions}

    # Limites x communes
    all_min = min((df["Alpha"].min() for df in dfs.values()), default=0)
    all_max = max((df["Alpha"].max() for df in dfs.values()), default=1)
    x_min = min(-0.05, float(all_min) - 0.05)
    x_max = max(1.0,  float(all_max) + 0.05)

    # Subplots
    fig = make_subplots(
        rows=len(questions),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=tuple(questions)
    )

    for i, q in enumerate(questions, start=1):
        df = dfs[q]
        fig.add_trace(
            go.Bar(
                orientation="h",
                y=df["Label"],
                x=df["Alpha"],
                hovertemplate="<b>%{y}</b><br>Alpha: %{x:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=i, col=1
        )

        # Ajuste marges latérales
        fig.update_yaxes(title_text="", automargin=True, row=i, col=1)

        # Gérer l'affichage de l'axe X
        if i < len(questions):
            fig.update_xaxes(showticklabels=False, title_text="", row=i, col=1)
        else:
            # dernier subplot → on affiche ticks + titre
            fig.update_xaxes(
                showticklabels=True,
                title_text="Alpha",
                row=i, col=1
            )

    # Lignes de repère globales
    shapes = []
    seuils = [(0.667, "dot", "Acceptable"), (0.800, "dash", "Bon")]
    for v, dashed, _label in seuils:
        shapes.append(
            dict(
                type="line",
                x0=v, x1=v,
                y0=0, y1=1,
                xref="x",       # axe x partagé (x1)
                yref="paper",   # toute la hauteur de la figure
                line=dict(width=1.5, dash=dashed, color="#2A3F5F"),
            )
        )

    # Annotations pour les seuils (sans écraser les titres des subplots)
    annotations = list(fig.layout.annotations)  # garder les sous-titres existants
    for v, _dashed, label in seuils:
        annotations.append(
            dict(
                x=v, y=0.02,
                xref="x", yref="paper",
                text=label,
                showarrow=False,
                textangle=-90,
                xanchor="left",
                yanchor="bottom",
                font=dict(size=12, color="#2A3F5F")
            )
        )

    # Mise en forme générale
    fig.update_layout(
        title=titre,
        title_font=dict(size=14, family="Arial", color="black"),
        xaxis=dict(range=[x_min, x_max], zeroline=True, zerolinecolor="#ddd"),
        shapes=shapes,
        annotations=annotations,  # combine sous-titres + labels seuils
        bargap=0.2,
        margin=dict(l=160, r=40, t=80, b=40),
        height=max(max_height, max_height * len(questions)),
        width=width
    )

    # Police des sous-titres uniquement
    for ann in fig['layout']['annotations']:
        if ann['text'] in questions:  # si c'est un sous-titre de subplot
            ann['font'] = dict(size=12, family="Arial", color="gray")
            
    
    return fig