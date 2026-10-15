"""NLP Sentiment Analysis — model definitions.

Two model tracks:
1. Classical baseline: TF-IDF + Logistic Regression
2. Transformer: HuggingFace DistilBERT (fine-tuned or zero-shot)
3. Topic modelling: BERTopic on negative reviews

If the transformer deps (torch, transformers) are not installed, the code
falls back to the TF-IDF baseline only.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_SEED = 42
TARGET_COL = "sentiment"
TEXT_COL = "text"
CLASSES = ["positive", "neutral", "negative"]


def split_data(df: pd.DataFrame, test_size: float = 0.20, seed: int = RANDOM_SEED):
    X = df[TEXT_COL].values
    y = df[TARGET_COL].values
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def build_tfidf_pipeline() -> Pipeline:
    """Classical baseline: TF-IDF (1-2 grams) + Logistic Regression."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_SEED, n_jobs=-1,
        )),
    ])


def evaluate_classifier(y_true, y_pred, y_proba=None) -> dict:
    """Compute per-class metrics + macro F1 + confusion matrix."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES).tolist()
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "per_class": {k: v for k, v in report.items() if k in CLASSES},
        "confusion_matrix": cm,
    }


# ────────────────────────────────────────────────────────────────────────────
# DistilBERT transformer (optional; requires torch + transformers)
# ────────────────────────────────────────────────────────────────────────────
def train_distilbert(X_train, y_train, X_test, y_test, epochs: int = 2):
    """Fine-tune DistilBERT on the training data; return predictions + model.

    Returns:
        (predictions, metrics, model_info) tuple
    """
    try:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification,
            TrainingArguments, Trainer, DataCollatorWithPadding,
        )
        from datasets import Dataset
        import torch
    except ImportError:
        return None, None, {"error": "transformers/datasets/torch not installed"}

    label2id = {lbl: i for i, lbl in enumerate(CLASSES)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(CLASSES),
        id2label=id2label, label2id=label2id,
    )

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    train_ds = Dataset.from_dict({"text": X_train.tolist(),
                                  "label": [label2id[l] for l in y_train]})
    test_ds  = Dataset.from_dict({"text": X_test.tolist(),
                                  "label": [label2id[l] for l in y_test]})
    train_ds = train_ds.map(tokenize_fn, batched=True)
    test_ds  = test_ds.map(tokenize_fn, batched=True)

    args = TrainingArguments(
        output_dir=str(Path(__file__).resolve().parents[1] / "artefacts" / "distilbert"),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=RANDOM_SEED,
        report_to="none",
        logging_steps=50,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=test_ds,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    preds_out = trainer.predict(test_ds)
    pred_ids = np.argmax(preds_out.predictions, axis=-1)
    pred_labels = [id2label[i] for i in pred_ids]
    metrics = evaluate_classifier(y_test, pred_labels)
    info = {"model_name": "distilbert-base-uncased", "epochs": epochs}
    return pred_labels, metrics, info


# ────────────────────────────────────────────────────────────────────────────
# Topic modelling: BERTopic on negative reviews (optional)
# ────────────────────────────────────────────────────────────────────────────
def topic_model_negative_reviews(df: pd.DataFrame, n_topics: int = 5):
    """Run BERTopic on negative reviews to surface recurring themes.

    Returns a dict: {topic_id: {'count': int, 'top_words': list[str]}}
    """
    try:
        from bertopic import BERTopic
    except ImportError:
        return {"error": "bertopic not installed; pip install bertopic"}
    neg = df[df[TARGET_COL] == "negative"][TEXT_COL].tolist()
    if len(neg) < 20:
        return {"error": f"Too few negative reviews ({len(neg)}) for topic modelling"}
    topic_model = BERTopic(nr_topics=n_topics, verbose=False)
    topics, _ = topic_model.fit_transform(neg)
    info = topic_model.get_topic_info()
    out = {}
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words = topic_model.get_topic(tid)
        out[tid] = {
            "count": int(row["Count"]),
            "top_words": [w[0] for w in words[:10]],
        }
    return out


def save_pipeline(pipeline, name: str, models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_pipeline(name: str, models_dir: Path):
    return joblib.load(models_dir / f"{name}.joblib")
