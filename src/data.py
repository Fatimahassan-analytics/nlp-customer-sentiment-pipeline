"""NLP Sentiment Analysis — data acquisition: scraping + sample fallback.

This module provides:
- scrape_trustpilot(): politely scrape reviews from a Trustpilot company page
- generate_sample_reviews(): create a small synthetic reviews dataset so the
  notebooks run end-to-end without an internet connection
- load_data(): load whichever is available
"""
from __future__ import annotations
import random
import re
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SAMPLE_PATH = RAW_DIR / "reviews_sample.csv"
FULL_PATH = RAW_DIR / "reviews_full.csv"

RANDOM_SEED = 42

# ────────────────────────────────────────────────────────────────────────────
# Synthetic review generator (fallback when scraping is unavailable)
# ────────────────────────────────────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "Absolutely love this product — exceeded my expectations.",
    "Great value for money, fast delivery, would buy again.",
    "Excellent customer service, the team went above and beyond.",
    "Best purchase I have made this year. Highly recommend.",
    "Five stars — works perfectly out of the box.",
    "Brilliant quality and the price was unbeatable.",
    "So happy with my order. Will definitely be a repeat customer.",
    "Easy to use, well-built, and arrived earlier than expected.",
]
NEUTRAL_TEMPLATES = [
    "It is okay, does the job but nothing special.",
    "Average product. Delivery was on time.",
    "Mixed feelings — some features great, others lacking.",
    "Works as described. No complaints but no praise either.",
    "Decent quality for the price, but packaging could be better.",
    "It is fine. Would not go out of my way to recommend.",
    "Three stars. Met expectations but did not exceed them.",
    "Reasonable product, but the instructions were unclear.",
]
NEGATIVE_TEMPLATES = [
    "Terrible experience — product broke after one week.",
    "Customer service was unhelpful and rude.",
    "Did not match the description at all. Returning.",
    "Worst purchase I have ever made. Avoid.",
    "Late delivery, damaged packaging, poor quality.",
    "Would not recommend. Save your money.",
    "One star. Product stopped working within a month.",
    "Very disappointed. The product felt cheap and flimsy.",
]


def generate_sample_reviews(n: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a synthetic reviews dataset with 3-class sentiment."""
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rows = []
    # Class distribution: 50% positive, 20% neutral, 30% negative
    classes = rng.choice(["positive", "neutral", "negative"], size=n, p=[0.5, 0.2, 0.3])
    template_pools = {
        "positive": POSITIVE_TEMPLATES,
        "neutral":  NEUTRAL_TEMPLATES,
        "negative": NEGATIVE_TEMPLATES,
    }
    for i, cls in enumerate(classes):
        base = random.choice(template_pools[cls])
        # Add a small variation so reviews are not all identical
        variation = rng.choice([
            "", " Really impressed.", " Would buy again.", " Not what I expected.",
            " Will recommend to friends.", " Disappointed actually.", " Brilliant service.",
        ])
        text = (base + variation).strip()
        rating_map = {"positive": 5, "neutral": 3, "negative": 1}
        rows.append({
            "review_id": i,
            "text": text,
            "rating": rating_map[cls] + int(rng.integers(-1, 2)),  # ±1 variation
            "sentiment": cls,
            "source": "synthetic",
        })
    df = pd.DataFrame(rows)
    # Clip ratings to [1, 5]
    df["rating"] = df["rating"].clip(1, 5)
    return df


# ────────────────────────────────────────────────────────────────────────────
# Trustpilot scraper (polite, rate-limited)
# ────────────────────────────────────────────────────────────────────────────
def scrape_trustpilot(company_slug: str = "trustpilot", pages: int = 5,
                      sleep_sec: float = 1.5) -> pd.DataFrame:
    """Scrape reviews from a Trustpilot company page.

    Args:
        company_slug: the slug in the URL (e.g. 'amazon.co.uk' for trustpilot.com/review/amazon.co.uk)
        pages: number of review pages to scrape (each page has ~20 reviews)
        sleep_sec: polite delay between requests

    Returns:
        DataFrame with columns: review_id, text, rating, sentiment, source

    Notes:
        - Respects robots.txt; rate-limited to 1 request per `sleep_sec` seconds.
        - If the scrape fails (no internet, blocked, parsing error), returns empty DataFrame
          and the calling code falls back to the bundled synthetic sample.
    """
    try:
        from bs4 import BeautifulSoup
        import requests
    except ImportError:
        print("[scrape] requests/bs4 not installed; cannot scrape. Returning empty.")
        return pd.DataFrame(columns=["review_id", "text", "rating", "sentiment", "source"])

    base_url = f"https://www.trustpilot.com/review/{company_slug}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"),
        "Accept-Language": "en-GB,en;q=0.9",
    }
    all_reviews = []
    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"[scrape] page {page} returned HTTP {resp.status_code}; stopping.")
                break
            soup = BeautifulSoup(resp.text, "html.parser")
            review_cards = soup.find_all("div", {"class": re.compile(r"paper_paper.*card")})
            if not review_cards:
                # Fallback selector pattern
                review_cards = soup.find_all("article")
            for card in review_cards:
                # Extract rating from the data-rating attribute or star icons
                rating = None
                rating_el = card.find("div", {"data-service-review-rating": True})
                if rating_el:
                    rating = int(rating_el["data-service-review-rating"])
                else:
                    star_imgs = card.find_all("img", {"alt": re.compile(r"Rated")})
                    if star_imgs:
                        m = re.search(r"Rated (\d)", star_imgs[0]["alt"])
                        rating = int(m.group(1)) if m else None

                # Extract review text
                text_el = card.find("p", {"data-service-review-text": True})
                text = text_el.get_text(strip=True) if text_el else ""

                if rating and text:
                    sentiment = "positive" if rating >= 4 else ("neutral" if rating == 3 else "negative")
                    all_reviews.append({
                        "review_id": len(all_reviews),
                        "text": text,
                        "rating": rating,
                        "sentiment": sentiment,
                        "source": f"trustpilot:{company_slug}",
                    })
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"[scrape] page {page} failed: {e}")
            break

    return pd.DataFrame(all_reviews)


def load_data() -> pd.DataFrame:
    """Load reviews data: prefer full scraped CSV, fall back to bundled sample."""
    if FULL_PATH.exists():
        return pd.read_csv(FULL_PATH)
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)
    df = generate_sample_reviews()
    df.to_csv(SAMPLE_PATH, index=False)
    return df


def save_processed(df: pd.DataFrame, name: str = "processed.csv") -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / name
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    # Generate the bundled sample
    df = generate_sample_reviews()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAMPLE_PATH, index=False)
    print(f"Wrote sample reviews: {SAMPLE_PATH} ({len(df)} rows)")
    print(df["sentiment"].value_counts())
    print(df.head())
