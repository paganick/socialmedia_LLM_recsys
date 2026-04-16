#!/usr/bin/env python3
"""
Build the post pool for LLM recommendation experiments.

For each platform, samples up to --pool-size posts satisfying the character
limit, assigns anonymous IDs, and writes the pool to outputs/pools/.

Outputs
-------
  outputs/pools/{platform}_pool.csv      post_id, user_id, text       (shareable)
  outputs/pools/{platform}_id_map.csv    user_id → username            (internal)
  outputs/pools/twitter_authors.csv      user_id, bio                  (shareable)

The id_map files are for internal use only (they link anonymous user_ids back
to real usernames, needed by infer_demographics.py).

Usage
-----
    python build_pool.py
    python build_pool.py --pool-size 5000 --max-chars 300 --seed 42
    python build_pool.py --datasets twitter bluesky
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

DATASET_PATHS = {
    "twitter": Path("datasets/twitter/posts.pkl"),
    "bluesky":  Path("datasets/bluesky/posts.pkl"),
    "reddit":   Path("datasets/reddit/posts.pkl"),
}

TWITTER_BIOS = Path("datasets/twitter/user_bios.csv")
OUT_DIR = Path("outputs/pools")


def load_posts(dataset: str, max_chars: int) -> pd.DataFrame:
    """Load posts from posts.pkl, normalise columns, filter by length."""
    path = DATASET_PATHS[dataset]
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    df = pd.read_pickle(path)

    text_col = "message" if "message" in df.columns else "text"
    username_col = next((c for c in ("username", "user_id") if c in df.columns), None)
    if username_col is None:
        print(f"ERROR: cannot find a username column in the {dataset} dataset.")
        sys.exit(1)

    df = df.rename(columns={text_col: "text", username_col: "username"})

    if "training" in df.columns:
        df = df[df["training"] == 1]

    before = len(df)
    df = df[df["text"].astype(str).str.len() <= max_chars].copy()
    print(f"  {len(df):,} posts after length filter (removed {before - len(df):,})")

    return df[["username", "text"]].dropna(subset=["text"])


def build_pool_and_map(df: pd.DataFrame) -> tuple:
    """
    Assign anonymous user_ids (user_00001 …) and sequential post_ids.

    Returns
    -------
    pool   : DataFrame  post_id | user_id | text
    id_map : DataFrame  user_id | username
    """
    unique_users = sorted(df["username"].unique())
    user_map = {u: f"user_{i+1:05d}" for i, u in enumerate(unique_users)}

    df = df.copy().reset_index(drop=True)
    df["user_id"] = df["username"].map(user_map)
    df["post_id"] = [f"post_{i+1:05d}" for i in range(len(df))]

    pool   = df[["post_id", "user_id", "text"]]
    id_map = pd.DataFrame({"user_id": list(user_map.values()),
                           "username": list(user_map.keys())})
    return pool, id_map


def build_twitter_authors(id_map: pd.DataFrame) -> pd.DataFrame | None:
    """Join the id_map with user bios; return user_id + bio DataFrame."""
    if not TWITTER_BIOS.exists():
        print(f"  WARNING: {TWITTER_BIOS} not found — skipping twitter_authors.csv")
        return None

    bios = pd.read_csv(TWITTER_BIOS)[["username", "description"]].dropna(subset=["description"])
    bios = bios[bios["description"].str.strip() != ""]
    bios["username"] = bios["username"].str.lower()

    id_map_lower = id_map.copy()
    id_map_lower["username"] = id_map_lower["username"].str.lower()

    authors = id_map_lower.merge(bios, on="username", how="inner")
    return authors[["user_id", "description"]].rename(columns={"description": "bio"})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+",
                        default=["twitter", "bluesky", "reddit"],
                        choices=["twitter", "bluesky", "reddit"],
                        help="Platforms to process (default: all three)")
    parser.add_argument("--pool-size", type=int, default=5000,
                        help="Posts to sample per platform (default: 5000)")
    parser.add_argument("--max-chars", type=int, default=300,
                        help="Maximum post length in characters (default: 300)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        print(f"\n{'='*55}")
        print(f"Platform: {dataset.upper()}")

        df = load_posts(dataset, args.max_chars)

        if args.pool_size < len(df):
            df = df.sample(n=args.pool_size, random_state=args.seed).reset_index(drop=True)
            print(f"  Sampled {args.pool_size:,} posts")
        else:
            print(f"  Using all {len(df):,} posts (fewer than requested {args.pool_size:,})")

        pool, id_map = build_pool_and_map(df)

        pool.to_csv(OUT_DIR / f"{dataset}_pool.csv", index=False)
        id_map.to_csv(OUT_DIR / f"{dataset}_id_map.csv", index=False)
        print(f"  Saved pool:    {OUT_DIR}/{dataset}_pool.csv  ({len(pool):,} posts, {id_map['user_id'].nunique():,} users)")
        print(f"  Saved id map:  {OUT_DIR}/{dataset}_id_map.csv")

        if dataset == "twitter":
            authors = build_twitter_authors(id_map)
            if authors is not None:
                authors.to_csv(OUT_DIR / "twitter_authors.csv", index=False)
                print(f"  Saved authors: {OUT_DIR}/twitter_authors.csv  ({len(authors):,} users with bio)")

    print(f"\n✓ Done.")
    print(f"  Next step: python run_llm_recommendation.py --dataset twitter --provider anthropic")


if __name__ == "__main__":
    main()
