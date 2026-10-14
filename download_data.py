"""Scrape-or-load script for the NLP Sentiment project.

Usage:
    python download_data.py                 # use bundled synthetic sample
    python download_data.py --scrape        # scrape Trustpilot
    python download_data.py --scrape --company amazon.co.uk --pages 10
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def ensure_sample() -> None:
    sample_path = RAW_DIR / "reviews_sample.csv"
    if sample_path.exists():
        return
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data import generate_sample_reviews
    df = generate_sample_reviews()
    df.to_csv(sample_path, index=False)
    print(f"[OK] Wrote synthetic sample → {sample_path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrape", action="store_true",
                        help="Scrape Trustpilot reviews instead of using the synthetic sample.")
    parser.add_argument("--company", default="amazon.co.uk",
                        help="Trustpilot company slug (e.g. 'amazon.co.uk').")
    parser.add_argument("--pages", type=int, default=5,
                        help="Number of Trustpilot pages to scrape.")
    args = parser.parse_args()

    ensure_sample()

    target = RAW_DIR / "reviews_full.csv"
    if target.exists():
        print(f"[OK] {target} already exists; skipping download.")
        return

    if args.scrape:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from data import scrape_trustpilot
        print(f"[INFO] Scraping Trustpilot: company={args.company}, pages={args.pages}")
        df = scrape_trustpilot(company_slug=args.company, pages=args.pages)
        if len(df) > 0:
            df.to_csv(target, index=False)
            print(f"[OK] Scraped {len(df)} reviews → {target}")
        else:
            print("[WARN] Scraping returned 0 reviews. Using synthetic sample instead.")
    else:
        print("[INFO] Not scraping. Using bundled synthetic sample.")
        print("       To scrape real reviews, run: python download_data.py --scrape")


if __name__ == "__main__":
    main()
