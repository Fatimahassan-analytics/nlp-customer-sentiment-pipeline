"""Shared utilities for the NLP Sentiment project."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid")


def save_fig(name: str, fig_dir: Path = Path("artefacts")) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def plot_class_distribution(df: pd.DataFrame, target="sentiment"):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.countplot(data=df, x=target, palette="Set2", ax=ax,
                  order=["positive", "neutral", "negative"])
    ax.set_title("Sentiment class distribution")
    ax.set_ylabel("Number of reviews")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    return ax


def plot_review_length_distribution(df: pd.DataFrame, text_col="text", target="sentiment"):
    df = df.copy()
    df["length"] = df[text_col].str.len()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(data=df, x="length", hue=target, bins=40, palette="Set2", ax=ax,
                  kde=True, element="step")
    ax.set_title("Review length distribution by sentiment")
    ax.set_xlabel("Number of characters")
    return ax


def plot_top_words_per_class(df: pd.DataFrame, text_col="text", target="sentiment", top_n=15):
    from sklearn.feature_extraction.text import CountVectorizer
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    classes = ["positive", "neutral", "negative"]
    for ax, cls in zip(axes, classes):
        texts = df[df[target] == cls][text_col].tolist()
        if not texts:
            ax.set_title(f"{cls} (no data)"); continue
        vec = CountVectorizer(stop_words="english", max_features=top_n, ngram_range=(1, 1))
        X = vec.fit_transform(texts)
        freqs = X.toarray().sum(axis=0)
        words = vec.get_feature_names_out()
        order = np.argsort(freqs)[::-1]
        ax.barh(range(top_n), freqs[order][::-1], color="#216b91")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(np.array(words)[order][::-1])
        ax.set_title(f"Top-{top_n} words — {cls}")
        ax.set_xlabel("Frequency")
    return axes


def plot_confusion(cm: list, classes: list = None):
    classes = classes or ["positive", "neutral", "negative"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    return ax
