# LLMs as Content Curators: A Large-Scale Audit of Recommendation Bias Across Providers and Platforms

**Nicolò Pagan** (University of Zurich) · **Christopher Barrie** (New York University) · **Chris A. Bail** (Duke University) · **Petter Törnberg** (University of Amsterdam)

*Paper under review.*

---

## Overview

This repository contains the full pipeline and aggregated outputs for a large-scale audit of recommendation bias in LLM-based content curation. We systematically test how three major LLM providers (Anthropic Claude, OpenAI GPT, Google Gemini) rank social media posts across three platforms (Twitter/X, Bluesky, Reddit) under six prompt framings, measuring bias along content, stylistic, and demographic dimensions.

**Experimental design:** 3 platforms × 3 providers × 6 prompt styles × 100 trials = 5,400 recommendation sessions.

---

## Reproducing the Paper

> **Quick start:** Steps 4–5 are **fully reproducible** from the aggregated data already provided in `analysis_outputs/` — no raw data or API keys required.
>
> ```bash
> python compute_bias_metrics.py
> python generate_figures.py
> # → figures appear in analysis_outputs/visualizations/paper_plots_final/
> ```

Steps 1–3 require raw social media data that cannot be redistributed (platform terms of service), but the methodology is fully documented below.

### Prerequisites

**Option A — Conda environment:**
```bash
conda env create -f environment.yml
conda activate llm-bias-analysis
```

**Option B — Apptainer container** (recommended on HPC systems):
```bash
apptainer build llm-recsys.sif llm-recsys.def
apptainer exec --bind $(pwd):$(pwd) llm-recsys.sif python generate_figures.py
```
A convenience script that wraps all downstream steps is provided:
```bash
bash run_analysis.sh
```

---

## Pipeline

```
Step 1   build_pool.py               ← requires raw datasets (not released)
Step 2   run_llm_recommendation.py   ← requires pool + API keys
Step 3a  compute_text_features.py    ← requires raw datasets (not released)
Step 3b  infer_demographics.py       ← Twitter only; requires Ollama
──────────────────────────────────────────────────────────────────────────────
Step 4   compute_bias_metrics.py     ← fully reproducible from provided data
Step 5   generate_figures.py         ← fully reproducible from provided data
```

---

### Step 1 — Build Post Pool  *(requires raw data)*

Samples up to 5,000 posts per platform (max 300 characters), deterministically anonymises author IDs, and writes a shareable pool CSV.

```bash
python build_pool.py --datasets twitter bluesky reddit
```

Output: `outputs/pools/{dataset}_pool.csv`

---

### Step 2 — LLM Recommendation Experiments  *(requires pool + API keys)*

For each dataset × provider combination, runs 100 trials per prompt style (6 styles), each time sampling 100 posts and asking the LLM to rank and return the top 10.

```bash
# Set the relevant API key (see config.yaml.example)
export ANTHROPIC_API_KEY="sk-ant-..."

python run_llm_recommendation.py --dataset twitter --provider anthropic
# repeat for all dataset × provider combinations
```

Datasets: `twitter`, `bluesky`, `reddit`  
Providers: `anthropic` (Claude Sonnet 4.5), `openai` (GPT-4o-mini), `gemini` (Gemini 2.0 Flash)  
Prompt styles: `general`, `popular`, `engaging`, `informative`, `controversial`, `neutral`

Output: `outputs/experiments/{dataset}_{provider}_{model}/post_level_data.csv`

---

### Step 3a — Text Feature Computation  *(requires raw datasets)*

Computes per-post features (sentiment via VADER, topic via Cardiff NLP RoBERTa, toxicity via Detoxify, stylistic features) with a post-level cache to avoid redundant computation across experiments.

```bash
python compute_text_features.py --experiment-dir outputs/experiments/twitter_anthropic_claude-sonnet-4-5
# repeat for each experiment directory
```

Cache: `outputs/cache/{dataset}_features.parquet`

---

### Step 3b — Demographic Inference  *(Twitter only; requires Ollama)*

Infers gender, political leaning, and minority status for Twitter/X authors using two locally-deployed LLMs (`llama3.1:8b` and `mistral:v0.2` via [Ollama](https://ollama.com)). Labels require consensus between both models; disagreements are recorded as `unknown`.

```bash
ollama pull llama3.1:8b
ollama pull mistral:v0.2
python infer_demographics.py --experiment-dir outputs/experiments/twitter_anthropic_claude-sonnet-4-5
```

Cache: `analysis_outputs/inferred_attributes/twitter_llm_attributes_with_bio.csv`

---

### Step 4 — Bias Metrics  *(fully reproducible)*

Computes all bias statistics: Cohen's d, Cramér's V, directional bias, Random Forest feature importance (SHAP), and AUROC across all 54 conditions.

```bash
python compute_bias_metrics.py
```

Outputs:
- `analysis_outputs/pool_vs_recommended_summary.csv`
- `analysis_outputs/directional_bias_data.csv`
- `analysis_outputs/feature_importance_data.csv`

---

### Step 5 — Paper Figures  *(fully reproducible)*

Generates all paper figures directly from the aggregated CSVs in `analysis_outputs/`.

```bash
python generate_figures.py
```

Output: `analysis_outputs/visualizations/paper_plots_final/`

---

## Repository Structure

```
├── build_pool.py                  # Step 1: build anonymised post pools
├── run_llm_recommendation.py      # Step 2: run recommendation experiments
├── compute_text_features.py       # Step 3a: text feature extraction
├── infer_demographics.py          # Step 3b: demographic inference (Twitter)
├── compute_bias_metrics.py        # Step 4: bias statistics
├── generate_figures.py            # Step 5: paper figures
├── run_analysis.sh                # Convenience: run Steps 3a–5 via apptainer
│
├── utils/llm_client.py            # Unified LLM client (Anthropic/OpenAI/Gemini/Ollama)
├── features/text_features.py      # Text feature extraction library
├── data/loaders.py                # Dataset loading utilities
│
├── environment.yml                # Conda environment
├── llm-recsys.def                 # Apptainer container definition
├── config.yaml.example            # API key reference
│
└── analysis_outputs/
    ├── pool_vs_recommended_summary.csv
    ├── directional_bias_data.csv
    ├── feature_importance_data.csv
    └── visualizations/paper_plots_final/   # All paper figures
```

---

## Citation

```bibtex
@article{pagan2025llms,
  title   = {LLMs as Content Curators: A Large-Scale Audit of Recommendation Bias Across Providers and Platforms},
  author  = {Pagan, Nicol\`o and Barrie, Christopher and Bail, Chris A. and T{\"o}rnberg, Petter},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
