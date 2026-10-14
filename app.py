"""Streamlit app — NLP Sentiment Analysis.

Run locally:
    streamlit run app.py

The app lets the user paste any review text and get:
- TF-IDF baseline sentiment prediction with confidence
- (optional) DistilBERT prediction if the model has been fine-tuned
- Top-5 topics in negative reviews (from BERTopic)
"""
from __future__ import annotations
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from data import load_data  # noqa: E402
from model import (  # noqa: E402
    build_tfidf_pipeline, split_data, evaluate_classifier,
    save_pipeline, load_pipeline, topic_model_negative_reviews,
    CLASSES, RANDOM_SEED,
)

MODELS_DIR = Path(__file__).parent / "artefacts"
CACHED_MODEL_PATH = MODELS_DIR / "tfidf_pipeline.joblib"


@st.cache_resource(show_spinner="Training TF-IDF baseline on first run ...")
def get_or_train_tfidf():
    if CACHED_MODEL_PATH.exists():
        return load_pipeline("tfidf_pipeline", MODELS_DIR)
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = build_tfidf_pipeline()
    pipe.fit(X_train, y_train)
    save_pipeline(pipe, "tfidf_pipeline", MODELS_DIR)
    return pipe


def main():
    st.set_page_config(page_title="NLP Sentiment Analysis", page_icon=":speech_balloon:",
                       layout="wide")
    st.title("NLP Sentiment Analysis")
    st.caption("Classify customer reviews as positive / neutral / negative, and surface recurring themes in negative reviews.")

    model = get_or_train_tfidf()
    df = load_data()

    # Sidebar: model performance
    st.sidebar.header("Model performance")
    X_train, X_test, y_train, y_test = split_data(df)
    y_pred = model.predict(X_test)
    metrics = evaluate_classifier(y_test, y_pred)
    st.sidebar.metric("Macro F1",  f"{metrics['f1_macro']:.3f}")
    st.sidebar.metric("Weighted F1", f"{metrics['f1_weighted']:.3f}")
    st.sidebar.metric("Macro precision", f"{metrics['precision_macro']:.3f}")
    st.sidebar.metric("Macro recall",    f"{metrics['recall_macro']:.3f}")

    # Main area
    st.subheader("Predict sentiment of a review")
    user_text = st.text_area("Paste review text here:",
                            value="The product arrived on time and works perfectly. Highly recommend!",
                            height=120)
    if st.button("Predict") and user_text.strip():
        pred = model.predict([user_text])[0]
        proba = model.predict_proba([user_text])[0]
        # Find probability of predicted class
        classes = model.classes_
        pred_idx = list(classes).index(pred)
        confidence = proba[pred_idx]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted sentiment", pred)
            color = "🟢" if pred == "positive" else ("🟡" if pred == "neutral" else "🔴")
            st.metric("Confidence", f"{confidence:.1%}")
            st.write(f"Signal: {color}")
        with col2:
            st.bar_chart(pd.DataFrame({"probability": proba}, index=classes))

    st.divider()
    st.subheader("Topics in negative reviews (BERTopic)")
    if st.button("Run BERTopic on negative reviews"):
        with st.spinner("Fitting BERTopic..."):
            topics = topic_model_negative_reviews(df, n_topics=5)
        if isinstance(topics, dict) and "error" not in topics:
            rows = []
            for tid, info in topics.items():
                rows.append({"Topic": tid, "Count": info["count"],
                             "Top words": ", ".join(info["top_words"][:8])})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            msg = topics.get("error", "BERTopic not available")
            st.warning(f"Could not run BERTopic: {msg}")

    st.divider()
    st.subheader("Sample data")
    st.dataframe(df.head(20), use_container_width=True)


if __name__ == "__main__":
    main()
