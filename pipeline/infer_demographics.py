#!/usr/bin/env python3
"""
Infer Twitter Author Demographic Attributes using Local LLMs + User Bios
========================================================================

Infers gender, political leaning, and minority status for Twitter/X authors
using two locally-deployed LLMs (llama3.1:8b and mistral:v0.2 via Ollama).
Both models must independently agree on a label; disagreements are recorded
as 'unknown'. Up to 3 inference rounds are attempted, adding 15 posts per
round. Input: Twitter description field (user bio) + up to 20 recent posts.

When --experiment-dir is given, inference is restricted to the authors that
actually appear in that experiment's post_level_data.csv, and the agreed
attributes are merged back into that same CSV.

Output
------
  analysis_outputs/inferred_attributes/twitter_llm_attributes_with_bio.csv

Usage
-----
  # New workflow: restrict to a specific experiment
  python infer_demographics.py --experiment-dir outputs/experiments/twitter_anthropic_claude-sonnet-4-5

  # Legacy: process all twitter authors, apply to all twitter experiments
  python infer_demographics.py --n-messages 20 --phase infer
  python infer_demographics.py --phase apply
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.llm_client import OllamaClient

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET = 'twitter'

DEFAULT_LLAMA_MODEL   = 'llama3.1:8b'
DEFAULT_MISTRAL_MODEL = 'mistral:v0.2'

N_MESSAGES  = 15
STEP        = 15
MAX_ROUNDS  = 3
SEED        = 42

VALID_GENDER    = {'male', 'female', 'non-binary', 'unknown'}
VALID_POLITICAL = {'left', 'center-left', 'center', 'center-right', 'right', 'unknown'}
VALID_MINORITY  = {'yes', 'no', 'unknown'}

OUT_DIR         = Path('analysis_outputs/inferred_attributes')
OUT_FILE        = OUT_DIR / 'twitter_llm_attributes_with_bio.csv'
CHECKPOINT_FILE = OUT_DIR / 'twitter_llm_attributes_with_bio_checkpoint.csv'

BIO_FILE        = Path('datasets/twitter/user_bios.csv')

CHECKPOINT_EVERY = 50

# ============================================================================
# PROMPT
# ============================================================================

SYSTEM_PROMPT = """\
You are a social-science researcher analysing anonymous social media posts.
Given a user's profile bio and a set of their posts, infer three demographic
attributes ONLY from explicit or strongly implied evidence in the text.

Attributes to infer:
  gender            – one of: male, female, non-binary, unknown
  political_leaning – one of: left, center-left, center, center-right, right, unknown
  is_minority       – one of: yes, no, unknown
                      (minority = racial/ethnic minority in the US context)

Rules:
  - Use "unknown" whenever there is insufficient evidence. Do not guess.
  - Return ONLY a JSON object with exactly these three keys. No explanation.

Example output:
{"gender": "female", "political_leaning": "left", "is_minority": "unknown"}
"""


def build_prompt(messages: List[str], bio: Optional[str] = None) -> str:
    numbered = "\n".join(f'{i+1}. "{m}"' for i, m in enumerate(messages))
    bio_section = f'User bio: "{bio}"\n\n' if bio else ""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{bio_section}"
        f"Posts:\n{numbered}\n\n"
        "JSON output:"
    )

# ============================================================================
# RESPONSE PARSING
# ============================================================================

def _extract_json(text: str) -> Optional[dict]:
    text = re.sub(r'```(?:json)?', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{[^{}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_response(text: str) -> Dict[str, str]:
    defaults = {'gender': 'unknown', 'political_leaning': 'unknown', 'is_minority': 'unknown'}
    obj = _extract_json(text)
    if obj is None:
        return defaults

    def _clean(val, valid_set):
        v = str(val).strip().lower().replace('_', '-')
        return v if v in valid_set else 'unknown'

    return {
        'gender':            _clean(obj.get('gender', ''),            VALID_GENDER),
        'political_leaning': _clean(obj.get('political_leaning', ''), VALID_POLITICAL),
        'is_minority':       _clean(obj.get('is_minority', ''),       VALID_MINORITY),
    }

# ============================================================================
# AGREEMENT LOGIC
# ============================================================================

def compute_agreed(llama: Dict[str, str], mistral: Dict[str, str]) -> Dict[str, str]:
    return {
        'gender':            llama['gender']            if llama['gender']            == mistral['gender']            else 'unknown',
        'political_leaning': llama['political_leaning'] if llama['political_leaning'] == mistral['political_leaning'] else 'unknown',
        'is_minority':       llama['is_minority']       if llama['is_minority']       == mistral['is_minority']       else 'unknown',
    }

# ============================================================================
# DATA LOADING
# ============================================================================

def load_personas() -> pd.DataFrame:
    path = Path(f'datasets/{DATASET}/posts.pkl')
    if not path.exists():
        raise FileNotFoundError(f"personas.pkl not found at {path}")
    df = pd.read_pickle(path)
    author_col = next(
        (c for c in df.columns if c in ('username', 'user_id', 'author', 'author_id')),
        df.columns[0]
    )
    msg_col = next(
        (c for c in df.columns if c in ('message', 'text', 'content', 'tweet')),
        None
    )
    if msg_col is None:
        raise ValueError(f"Cannot find message column in {list(df.columns)}")
    df = df.rename(columns={author_col: 'author_id', msg_col: 'message'})
    return df[['author_id', 'message']].dropna(subset=['message'])


def load_bios() -> Dict[str, str]:
    if not BIO_FILE.exists():
        raise FileNotFoundError(f"Bio file not found at {BIO_FILE}")
    df = pd.read_csv(BIO_FILE)
    df = df[['username', 'description']].dropna(subset=['description'])
    df = df[df['description'].str.strip() != '']
    return dict(zip(df['username'].str.lower(), df['description'].str.strip()))


def get_all_messages_shuffled(group: pd.DataFrame, seed: int) -> List[str]:
    msgs = group['message'].dropna().astype(str).tolist()
    msgs = [m.strip() for m in msgs if len(m.strip()) > 10]
    if not msgs:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(msgs))
    return [msgs[i] for i in idx]

# ============================================================================
# PHASE 1 – INFERENCE
# ============================================================================

def run_infer(llama_model: str, mistral_model: str, n_messages: int,
              step: int, max_rounds: int, seed: int, ollama_host: str,
              author_ids: set = None):

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to Ollama...")
    llama_client   = OllamaClient(model=llama_model,   host=ollama_host, temperature=0.0, max_tokens=128)
    mistral_client = OllamaClient(model=mistral_model, host=ollama_host, temperature=0.0, max_tokens=128)

    available = llama_client.list_models()
    print(f"Available Ollama models: {available}")
    for name, model in [('Llama', llama_model), ('Mistral', mistral_model)]:
        if not any(model in m for m in available):
            print(f"\n[ERROR] {name} model '{model}' not found in Ollama.")
            print(f"        Pull it with:  ollama pull {model}")
            sys.exit(1)

    print(f"\nUsing models:  Llama={llama_model}  |  Mistral={mistral_model}")
    print(f"Messages/author: {n_messages} initial, +{step} per retry round, max {max_rounds} rounds  |  Seed: {seed}")

    print(f"\nLoading bios from: {BIO_FILE}")
    bios = load_bios()
    print(f"Bios loaded: {len(bios):,}")

    df = load_personas()
    authors = df['author_id'].unique()

    if author_ids is not None:
        authors = [a for a in authors if str(a) in author_ids]
        print(f"Restricting inference to {len(authors):,} authors present in the experiment.")

    bio_coverage = sum(1 for a in authors if str(a).lower() in bios)
    print(f"\nDataset: {DATASET.upper()}  |  Authors: {len(authors):,}  |  Total posts: {len(df):,}")
    print(f"Authors with bio: {bio_coverage}/{len(authors)} ({100*bio_coverage/len(authors):.1f}%)")

    results = []
    done_ids = set()

    # Treat OUT_FILE as a persistent cache across runs
    if OUT_FILE.exists():
        cached = pd.read_csv(OUT_FILE)
        results = cached.to_dict('records')
        done_ids = set(cached['author_id'].astype(str))
        print(f"Loaded {len(done_ids)} already-processed authors from cache ({OUT_FILE}).")

    # Also absorb any in-progress checkpoint not yet merged into OUT_FILE
    if CHECKPOINT_FILE.exists():
        ckpt = pd.read_csv(CHECKPOINT_FILE)
        for record in ckpt.to_dict('records'):
            if str(record['author_id']) not in done_ids:
                results.append(record)
                done_ids.add(str(record['author_id']))
        print(f"  (+{len(done_ids) - len(results):,} from checkpoint)")

    remaining = [a for a in authors if str(a) not in done_ids]

    ATTRS = ('gender', 'political_leaning', 'is_minority')

    for i, author_id in enumerate(tqdm(remaining, desc=DATASET)):
        group = df[df['author_id'] == author_id]
        author_seed = seed ^ hash(str(author_id)) & 0xFFFF
        all_messages = get_all_messages_shuffled(group, seed=author_seed)
        bio = bios.get(str(author_id).lower())

        if not all_messages:
            results.append({
                'author_id': author_id, 'dataset': DATASET, 'has_bio': bio is not None,
                'llama_gender': 'unknown', 'mistral_gender': 'unknown', 'agreed_gender': 'unknown',
                'llama_political': 'unknown', 'mistral_political': 'unknown', 'agreed_political': 'unknown',
                'llama_minority': 'unknown', 'mistral_minority': 'unknown', 'agreed_minority': 'unknown',
                'n_messages': 0, 'n_rounds': 0, 'agreement_rate': 1.0,
            })
            continue

        llama_final   = {a: 'unknown' for a in ATTRS}
        mistral_final = {a: 'unknown' for a in ATTRS}
        agreed_final  = {a: 'unknown' for a in ATTRS}
        unresolved = set(ATTRS)
        n_used = 0
        n_rounds = 0

        while True:
            n_used = min(n_messages if n_rounds == 0 else n_used + step, len(all_messages))
            messages = all_messages[:n_used]
            n_rounds += 1

            prompt = build_prompt(messages, bio=bio)

            try:
                llama_attr = parse_response(llama_client.generate(prompt))
            except Exception as e:
                print(f"\n  [WARN] Llama failed for {author_id} round {n_rounds}: {e}")
                llama_attr = {a: 'unknown' for a in ATTRS}

            try:
                mistral_attr = parse_response(mistral_client.generate(prompt))
            except Exception as e:
                print(f"\n  [WARN] Mistral failed for {author_id} round {n_rounds}: {e}")
                mistral_attr = {a: 'unknown' for a in ATTRS}

            newly_resolved = set()
            for attr in unresolved:
                if llama_attr[attr] == mistral_attr[attr]:
                    agreed_final[attr]  = llama_attr[attr]
                    llama_final[attr]   = llama_attr[attr]
                    mistral_final[attr] = mistral_attr[attr]
                    newly_resolved.add(attr)
            unresolved -= newly_resolved

            for attr in unresolved:
                llama_final[attr]   = llama_attr[attr]
                mistral_final[attr] = mistral_attr[attr]

            if not unresolved or n_rounds >= max_rounds or n_used >= len(all_messages):
                break

        n_agreed = sum(1 for a in ATTRS if llama_final[a] == mistral_final[a])

        results.append({
            'author_id':         author_id,
            'dataset':           DATASET,
            'has_bio':           bio is not None,
            'llama_gender':      llama_final['gender'],
            'mistral_gender':    mistral_final['gender'],
            'agreed_gender':     agreed_final['gender'],
            'llama_political':   llama_final['political_leaning'],
            'mistral_political': mistral_final['political_leaning'],
            'agreed_political':  agreed_final['political_leaning'],
            'llama_minority':    llama_final['is_minority'],
            'mistral_minority':  mistral_final['is_minority'],
            'agreed_minority':   agreed_final['is_minority'],
            'n_messages':        n_used,
            'n_rounds':          n_rounds,
            'agreement_rate':    n_agreed / 3,
        })

        if (i + 1) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_FILE, index=False)
    print(f"\n✓ Saved: {OUT_FILE}")

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()

    print(f"\n  Agreement rates:")
    for col, label in [('agreed_gender', 'gender'), ('agreed_political', 'political'), ('agreed_minority', 'minority')]:
        n_known = (result_df[col] != 'unknown').sum()
        n_agree = (result_df[f'llama_{label}'] == result_df[f'mistral_{label}']).sum()
        print(f"    {label:20s}: {n_agree}/{len(result_df)} agree ({100*n_agree/len(result_df):.1f}%),  "
              f"{n_known} have non-unknown agreed value")

# ============================================================================
# PHASE 2 – APPLY TO EXPERIMENT OUTPUTS
# ============================================================================

ATTR_MAP = {
    'agreed_gender':    'author_gender',
    'agreed_political': 'author_political_leaning',
    'agreed_minority':  'author_is_minority',
}


def apply_to_experiment_outputs(experiments_dir: Path = None,
                                experiment_dir: Path = None):
    """Apply inferred attributes to experiment CSV(s).

    Pass experiment_dir (specific experiment) or experiments_dir (top-level
    directory; all twitter experiments under it will be updated).
    """
    print(f"\n{'='*70}")
    print("APPLYING INFERRED ATTRIBUTES TO EXPERIMENT OUTPUTS")
    print(f"{'='*70}")

    if not OUT_FILE.exists():
        print(f"[SKIP] No attribute file found — run --phase infer first.")
        return

    attr_df = pd.read_csv(OUT_FILE, dtype={'author_id': str})
    attr_df = attr_df[['author_id'] + list(ATTR_MAP.keys())]

    # Translate real usernames → anonymous user_ids so the merge key matches
    # the experiment CSV (which uses user_id from the pool).
    id_map_path = Path('outputs/pools') / f'{DATASET}_id_map.csv'
    if id_map_path.exists():
        id_map = pd.read_csv(id_map_path, dtype=str)
        attr_df = attr_df.merge(id_map, left_on='author_id', right_on='username', how='inner')
        attr_df = attr_df.drop(columns=['author_id', 'username'])
        # 'user_id' (anon) is now the join key; rename to 'author_id' for the loop below
        attr_df = attr_df.rename(columns={'user_id': 'author_id'})
    else:
        print(f"[WARN] {id_map_path} not found — merging on raw usernames (may not match).")

    # Resolve which experiment directories to update
    if experiment_dir is not None:
        if not experiment_dir.exists():
            print(f"[WARN] Experiment directory not found: {experiment_dir}")
            return
        exp_dirs = [experiment_dir]
    else:
        if experiments_dir is None:
            experiments_dir = Path('outputs/experiments')
        if not experiments_dir.exists():
            print(f"[WARN] Experiments directory not found: {experiments_dir}")
            return
        exp_dirs = sorted(experiments_dir.glob(f'{DATASET}_*'))
        if not exp_dirs:
            print(f"[SKIP] No experiment outputs found for '{DATASET}' in {experiments_dir}")
            return

    print(f"\n{DATASET.upper()}: applying to {len(exp_dirs)} experiment(s)...")

    for exp_dir in exp_dirs:
        csv_path = exp_dir / 'post_level_data.csv'
        if not csv_path.exists():
            continue

        post_df = pd.read_csv(csv_path)
        author_col = next(
            (c for c in post_df.columns if c in ('user_id', 'username', 'author_id', 'author')),
            None
        )
        if author_col is None:
            print(f"  [WARN] Cannot identify author column in {csv_path}. Skipping.")
            continue

        post_df[author_col] = post_df[author_col].astype(str)
        merged = post_df.merge(
            attr_df.rename(columns={'author_id': author_col}),
            on=author_col, how='left', suffixes=('', '_new')
        )

        n_updated = 0
        for src_col, dst_col in ATTR_MAP.items():
            if src_col in merged.columns:
                merged[dst_col] = merged[src_col]
                n_updated += merged[src_col].notna().sum()
                merged.drop(columns=[src_col], inplace=True, errors='ignore')

        merged.to_csv(csv_path, index=False)
        print(f"  ✓ {exp_dir.name}: {n_updated} values updated")

    print(f"\n✓ Done applying attributes.")

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--llama-model', default=DEFAULT_LLAMA_MODEL)
    p.add_argument('--mistral-model', default=DEFAULT_MISTRAL_MODEL)
    p.add_argument('--n-messages', type=int, default=N_MESSAGES)
    p.add_argument('--step', type=int, default=STEP)
    p.add_argument('--max-rounds', type=int, default=MAX_ROUNDS)
    p.add_argument('--seed', type=int, default=SEED)
    p.add_argument('--ollama-host', default='http://localhost:11434')
    p.add_argument('--phase', choices=['infer', 'apply', 'both'], default='both')
    p.add_argument('--experiment-dir', default=None,
                   help="Path to a specific experiment directory (e.g. "
                        "outputs/experiments/twitter_anthropic_claude-sonnet-4-5). "
                        "When set, inference is restricted to authors in that experiment "
                        "and attributes are applied back to its post_level_data.csv.")
    p.add_argument('--experiments-dir', default='outputs/experiments',
                   help="(Legacy) Top-level experiments directory; all twitter "
                        "experiments under it will be updated. Ignored when "
                        "--experiment-dir is set.")
    return p.parse_args()


def main():
    args = parse_args()

    experiment_dir = Path(args.experiment_dir) if args.experiment_dir else None

    # When a specific experiment dir is given, restrict inference to its authors.
    # The experiment CSV uses anonymised user_ids; use the id_map to recover real usernames.
    author_ids = None
    if experiment_dir is not None:
        csv_path = experiment_dir / "post_level_data.csv"
        id_map_path = Path("outputs/pools") / f"{DATASET}_id_map.csv"
        if csv_path.exists() and id_map_path.exists():
            post_df = pd.read_csv(csv_path, dtype=str)
            id_map  = pd.read_csv(id_map_path, dtype=str)
            anon_ids = set(post_df["user_id"].unique())
            real_usernames = set(
                id_map.loc[id_map["user_id"].isin(anon_ids), "username"]
            )
            author_ids = real_usernames
            print(f"Resolved {len(author_ids):,} real usernames from {len(anon_ids):,} "
                  f"anonymous user_ids via {id_map_path}")
        elif not csv_path.exists():
            print(f"[WARN] {csv_path} not found — inference will cover all authors.")
        elif not id_map_path.exists():
            print(f"[WARN] {id_map_path} not found — inference will cover all authors.")

    if args.phase in ('infer', 'both'):
        run_infer(
            llama_model=args.llama_model,
            mistral_model=args.mistral_model,
            n_messages=args.n_messages,
            step=args.step,
            max_rounds=args.max_rounds,
            seed=args.seed,
            ollama_host=args.ollama_host,
            author_ids=author_ids,
        )

    if args.phase in ('apply', 'both'):
        apply_to_experiment_outputs(
            experiment_dir=experiment_dir,
            experiments_dir=Path(args.experiments_dir) if experiment_dir is None else None,
        )


if __name__ == '__main__':
    main()
