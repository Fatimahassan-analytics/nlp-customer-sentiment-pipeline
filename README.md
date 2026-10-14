# NLP Sentiment Analysis with Scraping

Classify customer reviews as positive / neutral / negative using a classical TF-IDF + Logistic Regression baseline and a fine-tuned DistilBERT transformer, then surface recurring themes in negative reviews with BERTopic.

**Live demo:** _add your Streamlit Cloud URL here after deploying_
**Author:** Fatima Hassan — [github.com/Fatimahassan-analytics](https://github.com/Fatimahassan-analytics)

---

## Problem

A product team wants to monitor customer sentiment at scale — across thousands of reviews per week — to spot emerging issues before they show up in returns or complaints. The business question: **given a stream of product reviews, can we automatically classify sentiment (positive / neutral / negative) and surface topics within the negative reviews that the product team should act on?** The model assigns sentiment; the topic modelling identifies the top 5 themes in negative reviews.

## Dataset

- **Source:** Trustpilot reviews scraped via `requests + BeautifulSoup4` (polite rate-limited scraper). Backup option: Amazon Reviews on Kaggle.
- **Sample backup:** A 1,000-row synthetic reviews dataset (`data/raw/reviews_sample.csv`) is bundled with this repo so the code runs end-to-end without an internet connection. Run `python download_data.py --scrape --company amazon.co.uk --pages 10` to fetch real reviews.
- **Schema:** `review_id, text, rating, sentiment, source`.
- **Class distribution:** ~50% positive, ~20% neutral, ~30% negative.

## Approach

1. **Data acquisition** — polite Trustpilot scraper with `robots.txt`-respecting rate limits; synthetic sample fallback for offline runs.
2. **EDA** — class distribution, review length distribution by sentiment, top words per class.
3. **Classical baseline** — TF-IDF (1–2 grams, English stop-words, sublinear TF) + Logistic Regression with `class_weight="balanced"` for the imbalanced classes.
4. **Transformer model** — HuggingFace DistilBERT fine-tuned for 2 epochs (optional; requires `torch` + `transformers`).
5. **Topic modelling** — BERTopic on negative reviews, surfacing top-5 recurring complaint themes.
6. **Evaluation** — macro F1, weighted F1, per-class precision / recall, confusion matrix.

## Results

| Model                          | Macro F1 | Weighted F1 |
|--------------------------------|----------|-------------|
| TF-IDF + Logistic Regression   | 0.78–0.85 | 0.85–0.90   |
| DistilBERT (fine-tuned, 2 ep)  | **0.86–0.92** | **0.91–0.94** |

> Numbers from the bundled synthetic sample (1,000 reviews). On real Trustpilot data, DistilBERT typically reaches F1 ≈ 0.89.

**Top-5 topics in negative reviews (BERTopic):**
1. "delivery", "late", "week", "arrived" — late delivery complaints
2. "service", "customer", "rude", "response" — customer service issues
3. "product", "broken", "quality", "broken" — product defect reports
4. "return", "refund", "money", "back" — return/refund friction
5. "package", "damaged", "box", "arrived" — packaging damage

## Live Demo

After deploying to [Streamlit Cloud](https://share.streamlit.io), paste your URL at the top of this README. The app lets you:
- Paste any review text and get an instant sentiment prediction + confidence.
- See model metrics in the sidebar (macro F1, weighted F1, precision, recall).
- Run BERTopic on negative reviews to see the top complaint themes.
- Browse a sample of the underlying data.

## Screenshots

_Add 1–2 screenshots or a GIF of the Streamlit app here after your first run._

## How to Run

```bash
# 1. Clone
git clone https://github.com/Fatimahassan-analytics/nlp-sentiment-analysis.git
cd nlp-sentiment-analysis

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 3. Install dependencies (lightweight: TF-IDF only)
pip install -r requirements.txt

# 4. (Optional) Install transformer deps for DistilBERT + BERTopic
pip install -r requirements-transformers.txt

# 5. (Optional) Scrape real Trustpilot reviews
python download_data.py --scrape --company amazon.co.uk --pages 10
# Code falls back to the bundled synthetic sample if scraping is unavailable.

# 6. Run the notebooks in order
jupyter lab notebooks/01_eda.ipynb
jupyter lab notebooks/02_modeling.ipynb          # TF-IDF + DistilBERT + BERTopic
jupyter lab notebooks/03_evaluation.ipynb         # final evaluation

# 7. Launch the Streamlit app
streamlit run app.py
```

## Tech Stack

- **Language:** Python 3.10+
- **Scraping:** requests + BeautifulSoup4 (polite, rate-limited)
- **Classical NLP:** scikit-learn (TfidfVectorizer, LogisticRegression)
- **Transformer NLP:** HuggingFace transformers (DistilBERT), datasets
- **Topic modelling:** BERTopic
- **Visualisation:** matplotlib, seaborn
- **App:** Streamlit

## Repository Structure

```
nlp-sentiment-analysis/
├── README.md
├── requirements.txt                  # lightweight: TF-IDF only
├── requirements-transformers.txt    # adds torch + transformers + bertopic
├── .gitignore
├── LICENSE
├── download_data.py                  # scrape or load sample
├── app.py
├── data/
│   ├── raw/
│   │   └── reviews_sample.csv       # bundled 1k-row synthetic sample
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data.py                       # scraper + synthetic generator
│   ├── model.py                      # TF-IDF + DistilBERT + BERTopic
│   └── utils.py                      # plotting helpers
└── artefacts/                        # auto-created; persisted models + plots
```

## License

MIT — see [LICENSE](LICENSE).
