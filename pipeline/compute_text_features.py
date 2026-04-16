#!/usr/bin/env python3
"""
Compute text features for posts in an experiment output, with caching.

Features are cached per platform in outputs/cache/{dataset}_features.parquet,
keyed on post_id. On each run only posts missing from the cache are computed;
cached results are reused immediately. The enriched experiment CSV is written
back in place.

Features computed per post:
  - sentiment_polarity, sentiment_subjectivity  (VADER)
  - primary_topic, polarization_score           (Cardiff NLP RoBERTa)
  - has_emoji, has_hashtag, has_mention, has_url (regex)
  - avg_word_length                              (character-level)
  - toxicity                                    (Detoxify)

Input:  outputs/experiments/{experiment_dir}/post_level_data.csv
Cache:  outputs/cache/{dataset}_features.parquet
Output: outputs/experiments/{experiment_dir}/post_level_data.csv  (enriched in place)

Usage
-----
    python compute_text_features.py \\
        --experiment-dir outputs/experiments/twitter_anthropic_claude-sonnet-4-5
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from features.text_features import infer_tweet_metadata

CACHE_DIR = Path("outputs/cache")


def infer_dataset(exp_dir: Path) -> str:
    for dataset in ("twitter", "bluesky", "reddit"):
        if exp_dir.name.startswith(dataset):
            return dataset
    raise ValueError(f"Cannot infer platform from experiment directory name: {exp_dir.name}")


def load_cache(dataset: str) -> pd.DataFrame:
    path = CACHE_DIR / f"{dataset}_features.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["post_id"])


def save_cache(dataset: str, cache: pd.DataFrame):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(CACHE_DIR / f"{dataset}_features.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True,
                        help="Path to experiment directory containing post_level_data.csv")
    args = parser.parse_args()

    exp_dir  = Path(args.experiment_dir)
    csv_path = exp_dir / "post_level_data.csv"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.")
        print(f"       Run first:  python run_llm_recommendation.py ...")
        sys.exit(1)

    dataset = infer_dataset(exp_dir)

    print(f"Loading experiment data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows, {df['post_id'].nunique():,} unique post_ids")

    # ------------------------------------------------------------------ cache
    cache = load_cache(dataset)
    cached_ids = set(cache["post_id"]) if not cache.empty else set()

    unique_posts = df[["post_id", "text"]].drop_duplicates("post_id")
    missing = unique_posts[~unique_posts["post_id"].isin(cached_ids)]

    if missing.empty:
        print(f"All {len(unique_posts):,} posts already in cache — skipping computation.")
    else:
        print(f"\nCache: {len(cached_ids):,} posts already computed, "
              f"{len(missing):,} new posts to process.")
        features_df = infer_tweet_metadata(
            missing,
            text_column="text",
            sentiment_method="vader",
            topic_method="roberta",
            include_gender=False,
            include_political=False,
        )
        # Keep post_id and feature columns only (drop the text copy)
        feature_cols = [c for c in features_df.columns if c not in ("text",)]
        new_rows = features_df[feature_cols].copy()

        cache = pd.concat([cache, new_rows], ignore_index=True)
        cache = cache.drop_duplicates(subset=["post_id"], keep="last")
        save_cache(dataset, cache)
        print(f"  Cache updated: {len(cache):,} posts total → "
              f"{CACHE_DIR}/{dataset}_features.parquet")

    # ----------------------------------------------- merge into experiment CSV
    feature_cols = [c for c in cache.columns if c != "post_id"]
    # Drop any stale feature columns before merging
    df = df.drop(columns=[c for c in feature_cols if c in df.columns], errors="ignore")
    df = df.merge(cache[["post_id"] + feature_cols], on="post_id", how="left")

    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved enriched data ({len(df):,} rows, {len(df.columns)} columns) "
          f"to {csv_path}")
    print(f"  Next step: python compute_bias_metrics.py")


if __name__ == "__main__":
    main()
