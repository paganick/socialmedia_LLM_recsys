#!/usr/bin/env python3
"""
generate_figures.py — Single entry-point for all paper figures.

Reads from analysis_outputs/ (provided) and writes figures to:
    analysis_outputs/visualizations/paper_plots_final/

Switch
------
Set WITH_DEMOGRAPHICS = True  to include author demographic features
                               (Twitter/X only, LLM-inferred with user bios).
Set WITH_DEMOGRAPHICS = False to exclude them (all-platform plots only).

Usage
-----
    python generate_figures.py

Figures generated
-----------------
01   Aggregated R² bar plot (all features, ordered by magnitude)
02   Bias-by-prompt R² heatmap
03   Normalized bias-by-prompt heatmap (z-score within features)
04   Demographic directional bias heatmap       [only if WITH_DEMOGRAPHICS]
05   Content/safety bias by prompt × model heatmap
06   Content/safety directional bias bar charts by model
       06_polarization_score_by_model.png
       06_sentiment_polarity_by_model.png
       06_toxicity_by_model.png
07   Feature importance by model (SHAP, absolute values)
08a  Primary topic bias heatmap by dataset × model
08b  Primary topic bias heatmap by dataset × prompt
08c  Primary topic bias heatmap by dataset × model × prompt
09a  Avg word length directional bias by dataset / model / prompt
09b  Polarization directional bias by dataset / model / prompt
09c  Sentiment polarity directional bias by dataset / model / prompt
09d  Toxicity directional bias by dataset / model / prompt
10   Demographic bias by model                  [only if WITH_DEMOGRAPHICS]
       10_demo_bias_gender_by_model.png
       10_demo_bias_political_leaning_by_model.png
       10_demo_bias_is_minority_by_model.png

CSV data files saved alongside each figure for paper write-up.
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from pathlib import Path

# ============================================================================
# *** MAIN SWITCH ***
# ============================================================================

WITH_DEMOGRAPHICS = True   # ← set False to exclude author features

# ============================================================================
# PATHS — resolved after argument parsing in _init_paths()
# ============================================================================

def _init_paths(base_dir: Path):
    global BASE, ANALYSIS, OUT, SUMMARY_CSV, DIR_BIAS_CSV, IMPORTANCE_CSV, INFERRED_BIO_CSV
    BASE     = base_dir
    ANALYSIS = BASE / "analysis_outputs"
    OUT      = ANALYSIS / "visualizations" / "paper_plots_final_no_demographics" if not WITH_DEMOGRAPHICS else ANALYSIS / "visualizations" / "paper_plots_final"
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_CSV      = ANALYSIS / "pool_vs_recommended_summary.csv"
    DIR_BIAS_CSV     = ANALYSIS / "directional_bias_data.csv"
    IMPORTANCE_CSV   = ANALYSIS / "feature_importance_data.csv"
    INFERRED_BIO_CSV = ANALYSIS / "inferred_attributes" / "twitter_llm_attributes_with_bio.csv"

# Parse --base-dir early so path globals are set before any function uses them
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--base-dir", type=Path, default=Path(__file__).parent)
_args, _ = _parser.parse_known_args()
_init_paths(_args.base_dir)

# ============================================================================
# SHARED CONSTANTS
# ============================================================================

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 150
plt.rcParams["savefig.dpi"] = 300

AUTHOR_FEATURES      = ["author_gender", "author_political_leaning", "author_is_minority"]
DEMOGRAPHIC_DATASETS = ["twitter"]

FEATURES_ALL = {
    "author":       ["author_gender", "author_political_leaning", "author_is_minority"],
    "text_metrics": ["avg_word_length"],
    "sentiment":    ["sentiment_polarity", "sentiment_subjectivity"],
    "style":        ["has_emoji", "has_hashtag", "has_mention", "has_url"],
    "content":      ["polarization_score", "primary_topic"],
    "toxicity":     ["toxicity"],
}

FEATURE_DISPLAY = {
    "author_gender":            "Author: Gender",
    "author_political_leaning": "Author: Political Leaning",
    "author_is_minority":       "Author: Is Minority",
    "avg_word_length":          "Text: Avg Word Length",
    "polarization_score":       "Content: Polarization Score",
    "primary_topic":            "Content: Primary Topic",
    "sentiment_polarity":       "Sentiment: Polarity",
    "sentiment_subjectivity":   "Sentiment: Subjectivity",
    "has_emoji":                "Style: Has Emoji",
    "has_hashtag":              "Style: Has Hashtag",
    "has_mention":              "Style: Has Mention",
    "has_url":                  "Style: Has URL",
    "toxicity":                 "Toxicity: Toxicity",
}

CATEGORY_COLORS = {
    "author":       ["#8B4513", "#A0522D", "#CD853F"],
    "text_metrics": ["#1E90FF"],
    "content":      ["#32CD32", "#3CB371"],
    "sentiment":    ["#FFD700", "#FFA500"],
    "style":        ["#9370DB", "#8A2BE2", "#9400D3", "#9932CC"],
    "toxicity":     ["#DC143C", "#B22222"],
}

PROVIDER_ORDER  = ["anthropic", "openai", "gemini"]
PROVIDER_LABELS = {
    "anthropic": "Claude Sonnet 4.5",
    "openai":    "GPT-4o-mini",
    "gemini":    "Gemini 2.0 Flash",
}

DATASET_ORDER  = ["twitter", "bluesky", "reddit"]
DATASET_LABELS = {"twitter": "Twitter/X", "bluesky": "Bluesky", "reddit": "Reddit"}
DATASET_COLORS = {"bluesky": "#2166AC", "reddit": "#D6604D", "twitter": "#333333"}

PROMPT_ORDER  = ["neutral", "general", "popular", "engaging", "informative", "controversial"]
PROMPT_LABELS = {p: p.capitalize() for p in PROMPT_ORDER}

DIVG_COLORS = [
    "#2166AC", "#4393C3", "#92C5DE", "#D1E5F0", "#F7F7F7",
    "#FFFFFF",
    "#FEE0D2", "#FCBBA1", "#FC9272", "#FB6A4A", "#DE2D26",
]
CMAP_DIVG = LinearSegmentedColormap.from_list("diverging", DIVG_COLORS, N=256)
CMAP_WR   = LinearSegmentedColormap.from_list(
    "white_red",
    ["#FFFFFF", "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272",
     "#FB6A4A", "#EF3B2C", "#CB181D", "#A50F15", "#67000D"],
    N=256,
)

RQ3_METRICS = {
    "polarization_score": {
        "short_name": "Polarization",
        "ylabel":     "Polarization Bias\n(Recommended − Pool)",
    },
    "sentiment_polarity": {
        "short_name": "Sentiment",
        "ylabel":     "Sentiment Polarity Bias\n(Recommended − Pool)",
    },
    "toxicity": {
        "short_name": "Toxicity",
        "ylabel":     "Toxicity Bias\n(Recommended − Pool)",
    },
}

TOPIC_DISPLAY = {
    "news_&_social_concern":    "News &\nSocial Concern",
    "diaries_&_daily_life":     "Diaries &\nDaily Life",
    "sports":                   "Sports",
    "business_&_entrepreneurs": "Business &\nEntrepreneurs",
    "celebrity_&_pop_culture":  "Celebrity &\nPop Culture",
    "film_tv_&_video":          "Film, TV\n& Video",
}
TOP_N_TOPICS = 3

# ============================================================================
# SHARED HELPERS
# ============================================================================

def fmt(feature):
    return FEATURE_DISPLAY.get(feature, feature.replace("_", " ").title())

def get_category(feature):
    for cat, feats in FEATURES_ALL.items():
        if feature in feats:
            return cat
    return "other"

def get_color(feature, idx=0):
    colors = CATEGORY_COLORS.get(get_category(feature), ["#888888"])
    return colors[idx % len(colors)]

def to_r2(row):
    if pd.isna(row["bias"]) or pd.isna(row["metric"]):
        return np.nan
    v = abs(row["bias"])
    if row["metric"] == "Cohen's d":
        return (v ** 2) / (v ** 2 + 4)
    elif row["metric"] == "Cramér's V":
        return v ** 2
    return np.nan

def load_summary(with_demographics):
    df = pd.read_csv(SUMMARY_CSV)
    df["r_squared"] = df.apply(to_r2, axis=1)
    df = df.dropna(subset=["r_squared"])
    if with_demographics:
        mask = (~df["feature"].isin(AUTHOR_FEATURES)) | \
               (df["feature"].isin(AUTHOR_FEATURES) & df["dataset"].isin(DEMOGRAPHIC_DATASETS))
    else:
        mask = ~df["feature"].isin(AUTHOR_FEATURES)
    return df[mask]

def make_r2_annot(pivot_r2, pivot_sig, mean_row_name="Mean Across Features"):
    annot = np.empty_like(pivot_r2, dtype=object)
    for i in range(pivot_r2.shape[0]):
        rn = pivot_r2.index[i]
        for j in range(pivot_r2.shape[1]):
            val = pivot_r2.iloc[i, j]
            sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot[i, j] = ""
            elif rn == mean_row_name:
                annot[i, j] = f"{val:.3f}"
            elif sig > 0.75:
                annot[i, j] = f"{val:.3f}***"
            elif sig > 0.60:
                annot[i, j] = f"{val:.3f}**"
            elif sig > 0.50:
                annot[i, j] = f"{val:.3f}*"
            else:
                annot[i, j] = f"{val:.3f}"
    return annot

def _load_metric_bias(feature_name):
    df = pd.read_csv(DIR_BIAS_CSV)
    return df[
        (df["feature"] == feature_name) & (df["feature_type"] == "continuous")
    ][["provider", "dataset", "prompt_style", "directional_bias"]].copy()

# ============================================================================
# FIGURE 01: AGGREGATED R² BAR PLOT
# ============================================================================

def plot_01_aggregated_bar(comp_df):
    print("\n" + "="*70)
    print("FIGURE 01 — Aggregated R² bar plot")
    print("="*70)

    agg = comp_df.groupby("feature").agg(
        r_squared=("r_squared", "mean"),
        significant=("significant", "mean"),
    ).reset_index()
    agg["category"]        = agg["feature"].apply(get_category)
    agg["feature_display"] = agg["feature"].apply(fmt)
    agg = agg.sort_values("r_squared", ascending=False).reset_index(drop=True)

    colors, cat_counts = [], {}
    for _, row in agg.iterrows():
        idx = cat_counts.get(row["category"], 0)
        colors.append(get_color(row["feature"], idx))
        cat_counts[row["category"]] = idx + 1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(agg["feature_display"], agg["r_squared"],
           color=colors, edgecolor="black", alpha=0.8, linewidth=0.5)

    for i, row in agg.iterrows():
        marker = ("***" if row["significant"] > 0.75 else
                  "**"  if row["significant"] > 0.60 else
                  "*"   if row["significant"] > 0.50 else None)
        if marker:
            ax.text(i, row["r_squared"], marker, ha="center", va="bottom",
                    fontsize=12, fontweight="bold")

    ax.set_ylabel("Average R²", fontsize=12, fontweight="bold")
    ax.set_title("Average Bias per Feature (R²)\n(* p<0.05 >50%, ** >60%, *** >75%)",
                 fontweight="bold", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)

    legend_elements = []
    for cat in ["author", "text_metrics", "sentiment", "style", "content", "toxicity"]:
        feats = [f for f in FEATURES_ALL.get(cat, [])
                 if f not in AUTHOR_FEATURES or WITH_DEMOGRAPHICS]
        if not feats:
            continue
        label = cat.replace("_", " ").title()
        if cat == "author":
            label += " (Twitter/X only)"
        legend_elements.append(Patch(facecolor=get_color(feats[0], 0),
                                     edgecolor="black", label=label))
    ax.legend(handles=legend_elements, loc="upper right",
              title="Feature Category", fontsize=11, title_fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "01_aggregated_r2_bar_plot.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 01_aggregated_r2_bar_plot.png")

    agg[["feature", "feature_display", "category", "r_squared", "significant"]].to_csv(
        OUT / "01_aggregated_r2_bar_plot_data.csv", index=False)
    print("  ✓ 01_aggregated_r2_bar_plot_data.csv")

# ============================================================================
# FIGURE 02: BIAS BY PROMPT HEATMAP (R²)
# ============================================================================

def plot_02_bias_by_prompt(comp_df):
    print("\n" + "="*70)
    print("FIGURE 02 — Bias-by-prompt R² heatmap")
    print("="*70)

    agg_p = comp_df.groupby(["feature", "prompt_style"]).agg(
        r_squared=("r_squared", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a = comp_df.groupby("feature").agg(
        r_squared=("r_squared", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a["prompt_style"] = "Average"

    combined  = pd.concat([agg_p, agg_a], ignore_index=True)
    pivot_r2  = combined.pivot(index="feature", columns="prompt_style", values="r_squared")
    pivot_sig = combined.pivot(index="feature", columns="prompt_style", values="significant")

    col_order = PROMPT_ORDER + ["Average"]
    pivot_r2  = pivot_r2[col_order].sort_values("Average", ascending=False)
    pivot_sig = pivot_sig[col_order].reindex(pivot_r2.index)

    mean_vals = {c: pivot_r2[c].mean() for c in col_order}
    pivot_r2  = pd.concat([pivot_r2,
                            pd.Series(mean_vals, name="Mean Across Features").to_frame().T])
    pivot_sig = pd.concat([pivot_sig,
                            pd.Series({c: np.nan for c in col_order},
                                      name="Mean Across Features").to_frame().T])

    pivot_r2.index  = [fmt(f) if f != "Mean Across Features" else f for f in pivot_r2.index]
    pivot_sig.index = pivot_r2.index
    pivot_r2.columns  = [c.title() for c in pivot_r2.columns]
    pivot_sig.columns = pivot_r2.columns

    annot = make_r2_annot(pivot_r2, pivot_sig)

    cmap = LinearSegmentedColormap.from_list(
        "white_red",
        ["#FFFFFF", "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272",
         "#FB6A4A", "#EF3B2C", "#CB181D", "#99000D"],
        N=100,
    )
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_r2, annot=annot, fmt="", cmap=cmap,
                vmin=0, vmax=pivot_r2.max().max(), ax=ax,
                cbar_kws={"label": "R² (Variance Explained)"},
                linewidths=0.5, linecolor="lightgray",
                annot_kws={"fontsize": 15})
    ax.collections[0].colorbar.ax.yaxis.label.set_size(18)
    ax.collections[0].colorbar.ax.tick_params(labelsize=14)

    avg_col = pivot_r2.columns.get_loc("Average")
    ax.axvline(x=avg_col,   color="black", linewidth=3)
    ax.axvline(x=avg_col+1, color="black", linewidth=3)
    for i in range(len(pivot_r2)):
        ax.add_patch(plt.Rectangle((avg_col, i), 1, 1,
                                   fill=True, facecolor="lightgray", alpha=0.2,
                                   edgecolor="black", linewidth=3, zorder=0))
    ax.axhline(y=len(pivot_r2)-1, color="black", linewidth=3)

    demo_note = "(author demographics: Twitter/X only; " if WITH_DEMOGRAPHICS else "("
    ax.set_title(
        f"Bias by Prompt Style ($R^2$) — Aggregated across Datasets & Models\n"
        f"{demo_note}* p<0.05 >50%, ** >60%, *** >75%)",
        fontweight="bold", fontsize=15, pad=20)
    ax.set_xlabel("Prompt Style", fontsize=15, fontweight="bold")
    ax.set_ylabel("Feature",      fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(OUT / "02_bias_by_prompt_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 02_bias_by_prompt_heatmap.png")

    out_rows = []
    for feat_disp in pivot_r2.index:
        feat = next((k for k, v in FEATURE_DISPLAY.items() if v == feat_disp), feat_disp)
        row = {"feature": feat, "feature_display": feat_disp}
        for col in pivot_r2.columns:
            row[col.lower()] = pivot_r2.loc[feat_disp, col]
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(OUT / "02_bias_by_prompt_heatmap_data.csv", index=False)
    print("  ✓ 02_bias_by_prompt_heatmap_data.csv")

# ============================================================================
# FIGURE 03: NORMALIZED BIAS BY PROMPT HEATMAP
# ============================================================================

def plot_03_normalized_bias(comp_df):
    print("\n" + "="*70)
    print("FIGURE 03 — Normalized bias-by-prompt heatmap")
    print("="*70)

    agg_p = comp_df.groupby(["feature", "prompt_style"]).agg(
        bias=("bias", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a = comp_df.groupby("feature").agg(
        bias=("bias", "mean"), significant=("significant", "mean")
    ).reset_index()
    agg_a["prompt_style"] = "Average"

    combined  = pd.concat([agg_p, agg_a], ignore_index=True)
    pivot_b   = combined.pivot(index="feature", columns="prompt_style", values="bias")
    pivot_sig = combined.pivot(index="feature", columns="prompt_style", values="significant")
    pivot_b   = pivot_b[PROMPT_ORDER + ["Average"]]
    pivot_sig = pivot_sig[PROMPT_ORDER + ["Average"]]

    pivot_norm = pivot_b[PROMPT_ORDER].copy()
    for feature in pivot_norm.index:
        vals = pivot_b.loc[feature, PROMPT_ORDER].values.astype(float)
        mu, sd = vals.mean(), vals.std()
        pivot_norm.loc[feature, PROMPT_ORDER] = (vals - mu) / sd if sd > 0 else np.zeros_like(vals)

    avg_r2   = comp_df.groupby("feature")["r_squared"].mean()
    ordering = avg_r2.reindex(pivot_norm.index).sort_values(ascending=False).index
    pivot_norm = pivot_norm.reindex(ordering)
    pivot_sig_p = pivot_sig[PROMPT_ORDER].reindex(ordering)

    pivot_norm.index    = [fmt(f) for f in pivot_norm.index]
    pivot_sig_p.index   = pivot_norm.index
    pivot_norm.columns  = [c.title() for c in pivot_norm.columns]
    pivot_sig_p.columns = pivot_norm.columns

    annot = np.empty_like(pivot_norm, dtype=object)
    for i in range(pivot_norm.shape[0]):
        for j in range(pivot_norm.shape[1]):
            val = pivot_norm.iloc[i, j]
            sig = pivot_sig_p.iloc[i, j] if not pd.isna(pivot_sig_p.iloc[i, j]) else 0
            if pd.isna(val):
                annot[i, j] = ""
            elif sig > 0.75:
                annot[i, j] = f"{val:.1f}***"
            elif sig > 0.60:
                annot[i, j] = f"{val:.1f}**"
            elif sig > 0.50:
                annot[i, j] = f"{val:.1f}*"
            else:
                annot[i, j] = f"{val:.1f}"

    flat    = pivot_norm.values.flatten()
    flat    = flat[~np.isnan(flat)]
    max_abs = max(abs(flat.min()), abs(flat.max()))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_norm, annot=annot, fmt="", cmap=CMAP_DIVG,
                center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                cbar_kws={"label": "Normalized Bias (z-score)\n← Reduced | Enhanced →"},
                linewidths=0.5, linecolor="lightgray",
                annot_kws={"fontsize": 15})
    ax.collections[0].colorbar.ax.yaxis.label.set_size(16)
    ax.collections[0].colorbar.ax.tick_params(labelsize=15)

    demo_note = "(author demographics: Twitter/X only; " if WITH_DEMOGRAPHICS else "("
    ax.set_title(
        f"Normalized Bias by Prompt Style — Aggregated across Datasets & Models\n"
        f"{demo_note}red = enhanced, blue = reduced; * p<0.05 >50%, ** >60%, *** >75%)",
        fontweight="bold", fontsize=15, pad=20)
    ax.set_xlabel("Prompt Style", fontsize=16, fontweight="bold")
    ax.set_ylabel("Feature",      fontsize=16, fontweight="bold")
    ax.tick_params(labelsize=15)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(OUT / "03_bias_by_prompt_normalized_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ 03_bias_by_prompt_normalized_heatmap.png")

    out_rows = []
    for feat_disp in pivot_norm.index:
        feat = next((k for k, v in FEATURE_DISPLAY.items() if v == feat_disp), feat_disp)
        row = {"feature": feat, "feature_display": feat_disp}
        for col in pivot_norm.columns:
            row[f"{col.lower()}_normalized"] = pivot_norm.loc[feat_disp, col]
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(
        OUT / "03_bias_by_prompt_normalized_heatmap_data.csv", index=False)
    print("  ✓ 03_bias_by_prompt_normalized_heatmap_data.csv")

# ============================================================================
# FIGURE 04: DEMOGRAPHIC DIRECTIONAL BIAS (Twitter only)
# ============================================================================

def plot_04_demographics():
    print("\n" + "="*70)
    print("FIGURE 04 — Demographic directional bias (Twitter/X only)")
    print("="*70)

    dir_bias = pd.read_csv(DIR_BIAS_CSV)
    dir_bias = dir_bias[
        ~dir_bias["feature"].isin(AUTHOR_FEATURES) |
        dir_bias["dataset"].isin(DEMOGRAPHIC_DATASETS)
    ]

    inferred = pd.read_csv(INFERRED_BIO_CSV)
    col_map  = {
        "author_gender":            "agreed_gender",
        "author_political_leaning": "agreed_political",
        "author_is_minority":       "agreed_minority",
    }
    pool_rows = []
    for feature, col in col_map.items():
        vc = inferred[col].value_counts(normalize=True, dropna=False)
        vc.index = vc.index.map(lambda x: "unknown" if pd.isna(x) else x)
        for cat, prop in vc.items():
            pool_rows.append({"dataset": "twitter", "feature": feature,
                               "category": cat, "prop_pool": prop})
    pool_dist = pd.DataFrame(pool_rows)
    pool_dist.to_csv(OUT / "04_demographics_pool_distributions.csv", index=False)
    print("  ✓ 04_demographics_pool_distributions.csv")

    categorical_features = list(col_map.keys())
    normalized_rows = []
    for feature in categorical_features:
        fdata = dir_bias[dir_bias["feature"] == feature].copy()
        for (ds, prov, prompt), grp in fdata.groupby(["dataset", "provider", "prompt_style"]):
            bias_sum   = grp["directional_bias"].sum()
            correction = bias_sum / len(grp) if abs(bias_sum) > 1e-10 else 0
            grp = grp.copy()
            grp["directional_bias"] -= correction
            normalized_rows.append(grp)
    dir_bias = dir_bias[~dir_bias["feature"].isin(categorical_features)]
    dir_bias = pd.concat([dir_bias] + normalized_rows, ignore_index=True)

    sensitive_features = {
        "author_political_leaning": {
            "display_name": "Author Political Leaning",
            "categories":       ["left", "center-left", "center", "center-right", "right", "unknown"],
            "category_labels":  ["Left", "Center-Left", "Center", "Center-Right", "Right", "Unknown"],
        },
        "author_gender": {
            "display_name": "Author Gender",
            "categories":       ["female", "male", "non-binary", "unknown"],
            "category_labels":  ["Female", "Male", "Non-Binary", "Unknown"],
        },
        "author_is_minority": {
            "display_name": "Author Minority Status",
            "categories":       ["yes", "no", "unknown"],
            "category_labels":  ["Minority", "Non-Minority", "Unknown"],
        },
    }

    all_pivots = {}
    for feature, finfo in sensitive_features.items():
        fdata = dir_bias[(dir_bias["feature"] == feature) & (dir_bias["dataset"] == "twitter")]
        if len(fdata) == 0:
            continue

        mean_agg = fdata.groupby(["provider", "category"])["directional_bias"].mean().reset_index()
        std_agg  = fdata.groupby(["provider", "category"])["directional_bias"].std().reset_index()
        piv_m = mean_agg.pivot(index="provider", columns="category", values="directional_bias")
        piv_s = std_agg.pivot( index="provider", columns="category", values="directional_bias")

        avail = [c for c in finfo["categories"] if c in piv_m.columns]
        piv_m = piv_m[avail].reindex(PROVIDER_ORDER)
        piv_s = piv_s[avail].reindex(PROVIDER_ORDER)

        avg_row = pd.Series(piv_m.mean(axis=0), name="Average")
        std_row = pd.Series(piv_m.std(axis=0),  name="Average")
        piv_m   = pd.concat([piv_m, avg_row.to_frame().T])
        piv_s   = pd.concat([piv_s, std_row.to_frame().T])

        pool_for   = pool_dist[(pool_dist["feature"] == feature) & (pool_dist["dataset"] == "twitter")]
        cat_lookup = dict(zip(finfo["categories"], finfo["category_labels"]))
        col_labels = []
        for col in avail:
            disp = cat_lookup.get(col, col)
            pct  = pool_for[pool_for["category"] == col]["prop_pool"].values
            col_labels.append(f"{disp}\n({pct[0]*100:.1f}%)" if len(pct) > 0 else disp)

        piv_disp         = piv_m.copy()
        piv_disp.columns = col_labels
        piv_disp.index   = [
            p if p == "Average" else PROVIDER_LABELS.get(p, p)
            for p in piv_disp.index
        ]

        annot = np.empty_like(piv_disp, dtype=object)
        for i in range(piv_disp.shape[0]):
            for j in range(piv_disp.shape[1]):
                val = piv_disp.iloc[i, j]
                std = piv_s.iloc[i, j]
                if pd.isna(val):
                    annot[i, j] = ""
                elif pd.isna(std):
                    annot[i, j] = f"{val:.3f}"
                else:
                    annot[i, j] = f"{val:.3f}\n±{std:.3f}"

        all_vals = [x for x in piv_disp.values.flatten() if not pd.isna(x)]
        max_abs  = max(abs(min(all_vals)), abs(max(all_vals)))
        all_pivots[feature] = (piv_disp, piv_s, piv_m, avail, annot, max_abs, finfo)

        rows = []
        for p in PROVIDER_ORDER + ["Average"]:
            if p not in piv_m.index:
                continue
            row = {"dataset": "twitter", "provider": p}
            for cat in avail:
                row[cat]          = piv_m.loc[p, cat]
                row[f"{cat}_std"] = piv_s.loc[p, cat]
            rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / f"04_{feature}_twitter_data.csv", index=False)
        print(f"  ✓ 04_{feature}_twitter_data.csv")

    feats_ordered = [f for f in sensitive_features if f in all_pivots]
    width_ratios  = [len(all_pivots[f][3]) for f in feats_ordered]
    fig, axes = plt.subplots(1, len(feats_ordered), figsize=(20, 7),
                              gridspec_kw={"width_ratios": width_ratios})
    if len(feats_ordered) == 1:
        axes = [axes]

    fig.suptitle(
        "Author Demographic Directional Bias – Twitter/X (LLM-inferred with user bios; averaged over all prompt styles)",
        fontweight="bold", fontsize=18, y=1.02)

    cbar_label = "Directional Bias\n← Under | Over-represented →"
    for idx, feature in enumerate(feats_ordered):
        ax = axes[idx]
        piv_disp, piv_s, _, avail, annot, max_abs, finfo = all_pivots[feature]

        sns.heatmap(piv_disp, annot=annot, fmt="", cmap=CMAP_DIVG,
                    center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                    cbar=True, cbar_kws={"label": cbar_label, "shrink": 1.0},
                    linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 16})

        if ax.collections:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=17)
                cbar.set_label(cbar_label, fontsize=17, fontweight="bold")

        ax.axhline(y=len(piv_disp)-1, color="black", linewidth=2.5)
        ax.set_title(finfo["display_name"], fontsize=17, fontweight="bold", pad=12)
        ax.set_xlabel("Category", fontsize=17, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Model", fontsize=17, fontweight="bold")
            ax.tick_params(axis="y", labelsize=17)
            plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
            ax.tick_params(axis="y", left=False)
        ax.tick_params(axis="x", labelsize=17)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    plt.tight_layout(w_pad=3)
    fig.savefig(OUT / "04_demographics_directional_bias_heatmap.png",
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 04_demographics_directional_bias_heatmap.png")

# ============================================================================
# FIGURE 05: CONTENT/SAFETY COMBINED HEATMAP (prompt × model)
# ============================================================================

def plot_05_content_safety_heatmap():
    print("\n" + "="*70)
    print("FIGURE 05 — Content/safety heatmap (prompt × model)")
    print("="*70)

    fig, axes  = plt.subplots(1, 3, figsize=(20, 8.05))
    all_rows   = []

    for idx, (feat, minfo) in enumerate(RQ3_METRICS.items()):
        ax   = axes[idx]
        data = _load_metric_bias(feat)

        agg_m = data.groupby(["provider", "prompt_style"])["directional_bias"].mean().reset_index()
        piv_m = agg_m.pivot(index="prompt_style", columns="provider",
                             values="directional_bias")
        piv_m = piv_m.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)

        # Compute std before adding avg row/col (positional arrays for annotation)
        std_col_vals  = piv_m.std(axis=1).values          # std across models per prompt style
        std_row_vals  = piv_m.std(axis=0).values          # std across prompt styles per model
        overall_std   = float(np.nanstd(piv_m.values))    # overall std

        # Compute per-row average (across models) before adding avg row
        avg_col = piv_m.mean(axis=1)

        avg_row = pd.Series(piv_m.mean(axis=0), name="Average")
        piv_m   = pd.concat([piv_m, avg_row.to_frame().T])

        # Add average column (mean across models per prompt style + overall mean)
        avg_col["Average"] = avg_col.mean()
        piv_m["Average"]   = avg_col

        short = {p: PROVIDER_LABELS[p] for p in PROVIDER_ORDER}
        short["Average"] = "Average"
        piv_m.columns = [short.get(p, p) for p in piv_m.columns]
        piv_m.index   = [PROMPT_LABELS.get(p, p) for p in piv_m.index]

        n_rows, n_cols = piv_m.shape
        annot = np.empty_like(piv_m, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                val = piv_m.iloc[i, j]
                if pd.isna(val):
                    annot[i, j] = ""
                elif i == n_rows - 1 and j == n_cols - 1:   # corner cell
                    annot[i, j] = f"{val:.3f}\n±{overall_std:.3f}"
                elif i == n_rows - 1:                        # avg row
                    annot[i, j] = f"{val:.3f}\n±{std_row_vals[j]:.3f}"
                elif j == n_cols - 1:                        # avg col
                    annot[i, j] = f"{val:.3f}\n±{std_col_vals[i]:.3f}"
                else:
                    annot[i, j] = f"{val:.3f}"

        max_abs = max(abs(piv_m.min().min()), abs(piv_m.max().max()))
        sns.heatmap(piv_m, annot=annot, fmt="", cmap=CMAP_DIVG,
                    center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                    cbar=True,
                    cbar_kws={"label": minfo["ylabel"].replace("\n", " ")},
                    linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 16})
        ax.collections[0].colorbar.ax.yaxis.label.set_size(18)
        ax.collections[0].colorbar.ax.tick_params(labelsize=16)
        ax.axhline(y=len(piv_m)-1, color="black", linewidth=2.5)
        ax.axvline(x=len(piv_m.columns)-1, color="black", linewidth=2.5)
        ax.set_title(minfo["short_name"], fontweight="bold", fontsize=20, pad=12)
        ax.set_xlabel("Model", fontsize=18, fontweight="bold")
        ax.set_ylabel("Prompt Style" if idx == 0 else "", fontsize=18, fontweight="bold")
        ax.tick_params(labelsize=17)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
        if idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=15)

        for p_orig in PROVIDER_ORDER:
            for prompt in PROMPT_ORDER:
                val = data[(data["provider"] == p_orig) & (data["prompt_style"] == prompt)][
                    "directional_bias"].mean()
                all_rows.append({"feature": feat, "provider": p_orig,
                                  "provider_display": PROVIDER_LABELS[p_orig],
                                  "prompt_style": prompt,
                                  "prompt_display": PROMPT_LABELS[prompt],
                                  "directional_bias": val})

    fig.suptitle(
        "Content and Safety Directional Bias by Model and Prompt Style "
        "(Averaged across Bluesky, Reddit, Twitter/X)",
        fontweight="bold", fontsize=24, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "05_content_safety_bias_by_prompt_and_model_heatmap.png",
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 05_content_safety_bias_by_prompt_and_model_heatmap.png")

    pd.DataFrame(all_rows).to_csv(OUT / "05_content_safety_bias_heatmap_data.csv", index=False)
    print("  ✓ 05_content_safety_bias_heatmap_data.csv")

# ============================================================================
# FIGURE 06: CONTENT/SAFETY BAR CHARTS (model × dataset × prompt)
# ============================================================================

def plot_06_content_safety_bars():
    print("\n" + "="*70)
    print("FIGURE 06 — Content/safety bar charts by model × dataset")
    print("="*70)

    all_rows = []
    for feat, minfo in RQ3_METRICS.items():
        data  = _load_metric_bias(feat)
        y_min = data["directional_bias"].min()
        y_max = data["directional_bias"].max()
        y_rng = y_max - y_min
        y_lim = (y_min - 0.1*y_rng, y_max + 0.1*y_rng)

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        fig.suptitle(
            f"{minfo['short_name']} Directional Bias by Model and Dataset\n"
            "(Grouped by Prompt Style)",
            fontweight="bold", fontsize=16, y=0.98)

        for idx, prompt in enumerate(PROMPT_ORDER):
            ax    = axes[idx]
            pdata = data[data["prompt_style"] == prompt]
            x     = np.arange(len(PROVIDER_ORDER))
            bar_w = 0.25

            for ds_idx, ds in enumerate(DATASET_ORDER):
                vals = []
                for p in PROVIDER_ORDER:
                    sub = pdata[(pdata["provider"] == p) & (pdata["dataset"] == ds)][
                        "directional_bias"]
                    vals.append(sub.values[0] if len(sub) > 0 else 0)
                ax.bar(x + (ds_idx - 1)*bar_w, vals, bar_w,
                       label=DATASET_LABELS[ds],
                       color=DATASET_COLORS[ds], alpha=0.8,
                       edgecolor="black", linewidth=0.5)

            ax.axhline(y=0, color="black", linewidth=0.8, alpha=0.3)
            ax.set_title(PROMPT_LABELS[prompt], fontweight="bold", fontsize=13)
            ax.set_ylabel(minfo["ylabel"], fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels([PROVIDER_LABELS[p].split()[0] for p in PROVIDER_ORDER],
                               fontsize=10)
            ax.set_ylim(*y_lim)
            ax.grid(axis="y", alpha=0.3)
            if idx == 0:
                ax.legend(loc="upper left", fontsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / f"06_{feat}_by_model.png", bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  ✓ 06_{feat}_by_model.png")

        out = data.copy()
        out["feature"]          = feat
        out["provider_display"] = out["provider"].map(PROVIDER_LABELS)
        out["dataset_display"]  = out["dataset"].map(DATASET_LABELS)
        out["prompt_display"]   = out["prompt_style"].map(PROMPT_LABELS)
        all_rows.append(out)

    pd.concat(all_rows, ignore_index=True).to_csv(
        OUT / "06_content_safety_bars_data.csv", index=False)
    print("  ✓ 06_content_safety_bars_data.csv")

# ============================================================================
# FIGURE 07: FEATURE IMPORTANCE BY MODEL (SHAP)
# ============================================================================

def plot_07_feature_importance(with_demographics):
    print("\n" + "="*70)
    print("FIGURE 07 — Feature importance by model (SHAP, absolute)")
    print("="*70)

    df = pd.read_csv(IMPORTANCE_CSV)
    if "feature" in df.columns and "shap_importance" in df.columns:
        df_long = df[["feature", "provider", "shap_importance"]].copy()
    else:
        shap_cols = [c for c in df.columns if c.startswith("shap_") and c != "shap_file"]
        rows = []
        for _, row in df.iterrows():
            for feat in [c.replace("shap_", "") for c in shap_cols]:
                rows.append({"provider": row["provider"], "feature": feat,
                              "shap_importance": row[f"shap_{feat}"]})
        df_long = pd.DataFrame(rows)

    if not with_demographics:
        df_long = df_long[~df_long["feature"].isin(AUTHOR_FEATURES)]

    agg   = df_long.groupby(["feature", "provider"])["shap_importance"].mean().reset_index()
    pivot = agg.pivot_table(values="shap_importance", index="provider",
                            columns="feature", aggfunc="mean").reindex(PROVIDER_ORDER)

    avg_row = pd.Series(pivot.mean(axis=0), name="Average\n(across models)")
    pivot   = pd.concat([pivot, avg_row.to_frame().T])

    sorted_feats = pivot.loc["Average\n(across models)"].sort_values(ascending=False).index.tolist()
    pivot = pivot[sorted_feats]

    pivot.index   = [idx if "Average" in str(idx) else PROVIDER_LABELS.get(idx, idx)
                     for idx in pivot.index]
    pivot.columns = [fmt(f) for f in pivot.columns]

    annot = np.empty_like(pivot, dtype=object)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            annot[i, j] = "" if pd.isna(val) else f"{val:.3f}"

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(pivot, annot=annot, fmt="", cmap=CMAP_WR,
                vmin=0, vmax=pivot.max().max(), ax=ax,
                cbar_kws={"label": "SHAP Importance"},
                linewidths=0.5, linecolor="lightgray",
                annot_kws={"fontsize": 15})
    ax.axhline(y=len(pivot)-1, color="black", linewidth=2.5)
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(15)
    cbar.ax.tick_params(labelsize=15)

    ax.set_title("Feature Importance by Model\n(Aggregated across datasets & prompts)",
                 fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Feature",        fontsize=16, fontweight="bold")
    ax.set_ylabel("Model", fontsize=16, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=15)

    plt.tight_layout()
    fig.savefig(OUT / "07_feature_importance_by_model.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 07_feature_importance_by_model.png")

    pivot.to_csv(OUT / "07_feature_importance_by_model_data.csv")
    print("  ✓ 07_feature_importance_by_model_data.csv")

# ============================================================================
# FIGURES 08a/b/c: PRIMARY TOPIC DIRECTIONAL BIAS
# ============================================================================

def _topic_col_labels(pt, topics, ds, prov=None, prompt=None):
    mask = pt["dataset"] == ds
    if prov:
        mask &= pt["provider"] == prov
    if prompt:
        mask &= pt["prompt_style"] == prompt
    pool = pt[mask].groupby("category")["prop_pool"].mean()
    out  = []
    for t in topics:
        disp = TOPIC_DISPLAY.get(t, t.replace("_", " ").title())
        pct  = pool.get(t, np.nan)
        out.append(f"{disp}\n({pct*100:.1f}%)" if not np.isnan(pct) else disp)
    return out

def _make_topic_annot(piv, std=None):
    annot = np.empty_like(piv.values, dtype=object)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.isna(v):
                annot[i, j] = ""
            elif std is not None and not pd.isna(std.iloc[i, j]):
                annot[i, j] = f"{v:+.3f}\n±{std.iloc[i, j]:.3f}"
            else:
                annot[i, j] = f"{v:+.3f}"
    return annot

def _draw_topic_heatmap(ax, piv, annot, vmax, col_lbls, row_lbls,
                        show_cbar=False, show_ylabel=True,
                        ylabel="", title="", annot_fs=15):
    piv_disp         = piv.copy()
    piv_disp.columns = col_lbls
    piv_disp.index   = row_lbls
    im = sns.heatmap(piv_disp, annot=annot, fmt="", cmap=CMAP_DIVG,
                     center=0, vmin=-vmax, vmax=vmax, ax=ax,
                     linewidths=0.5, linecolor="gray",
                     cbar=show_cbar,
                     cbar_kws={"label": "Directional Bias", "shrink": 1.0} if show_cbar else {},
                     annot_kws={"fontsize": annot_fs})
    if show_cbar:
        im.collections[0].colorbar.ax.yaxis.label.set_size(15)
        im.collections[0].colorbar.ax.tick_params(labelsize=15)
    ax.set_title(title, fontweight="bold", fontsize=16, pad=16)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel if show_ylabel else "", fontsize=16, fontweight="bold")
    if not show_ylabel:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=15)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)

def _shared_vmax(pivots):
    vals = [np.nanmax(np.abs(p.values)) for p in pivots
            if not p.empty and not p.isna().all().all()]
    return max(vals) if vals else 0.1

def plot_08_topic_heatmaps():
    print("\n" + "="*70)
    print("FIGURES 08a/b/c — Primary topic directional bias")
    print("="*70)

    df = pd.read_csv(DIR_BIAS_CSV)
    pt = df[df["feature"] == "primary_topic"].copy()

    top_topics = (
        pt.groupby("category")["prop_pool"].mean()
        .sort_values(ascending=False)
        .head(TOP_N_TOPICS)
        .index.tolist()
    )
    print(f"  Top {TOP_N_TOPICS} topics: {top_topics}")
    pt = pt[pt["category"].isin(top_topics)].copy()

    csv_data = pt.groupby(["dataset", "provider", "prompt_style", "category"])[
        "directional_bias"].mean().reset_index()
    csv_data["category_display"] = csv_data["category"].map(
        lambda t: TOPIC_DISPLAY.get(t, t.replace("_", " ").title()))
    csv_data.to_csv(OUT / "08_topic_bias_data.csv", index=False)
    print("  ✓ 08_topic_bias_data.csv")

    # 08a: rows=models, cols=topics, panels=datasets
    pivots, stds = [], []
    for ds in DATASET_ORDER:
        sub  = pt[pt["dataset"] == ds]
        mean = sub.groupby(["provider", "category"])["directional_bias"].mean().reset_index()
        std  = sub.groupby(["provider", "category"])["directional_bias"].std().reset_index()
        piv  = mean.pivot(index="provider",  columns="category",
                          values="directional_bias").reindex(index=PROVIDER_ORDER, columns=top_topics)
        spiv = std.pivot( index="provider",  columns="category",
                          values="directional_bias").reindex(index=PROVIDER_ORDER, columns=top_topics)
        pivots.append(piv)
        stds.append(spiv)

    vmax = _shared_vmax(pivots)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for i, (ds, piv, spiv) in enumerate(zip(DATASET_ORDER, pivots, stds)):
        _draw_topic_heatmap(axes[i], piv, _make_topic_annot(piv, spiv), vmax,
                            col_lbls=_topic_col_labels(pt, top_topics, ds),
                            row_lbls=[PROVIDER_LABELS[p] for p in PROVIDER_ORDER],
                            show_cbar=(i == 2), show_ylabel=(i == 0),
                            ylabel="Model", title=DATASET_LABELS[ds])
    fig.suptitle("Primary Topic Directional Bias by Dataset and Model\n"
                 "(Averaged across Prompt Styles, ±SD shown in annotations)",
                 fontweight="bold", fontsize=18, y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "08a_topic_bias_by_dataset_model.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 08a_topic_bias_by_dataset_model.png")

    # 08b: rows=prompts, cols=topics, panels=datasets
    pivots, stds = [], []
    for ds in DATASET_ORDER:
        sub  = pt[pt["dataset"] == ds]
        mean = sub.groupby(["prompt_style", "category"])["directional_bias"].mean().reset_index()
        std  = sub.groupby(["prompt_style", "category"])["directional_bias"].std().reset_index()
        piv  = mean.pivot(index="prompt_style", columns="category",
                          values="directional_bias").reindex(index=PROMPT_ORDER, columns=top_topics)
        spiv = std.pivot( index="prompt_style", columns="category",
                          values="directional_bias").reindex(index=PROMPT_ORDER, columns=top_topics)
        pivots.append(piv)
        stds.append(spiv)

    vmax = _shared_vmax(pivots)
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for i, (ds, piv, spiv) in enumerate(zip(DATASET_ORDER, pivots, stds)):
        _draw_topic_heatmap(axes[i], piv, _make_topic_annot(piv, spiv), vmax,
                            col_lbls=_topic_col_labels(pt, top_topics, ds),
                            row_lbls=[PROMPT_LABELS[p] for p in PROMPT_ORDER],
                            show_cbar=(i == 2), show_ylabel=(i == 0),
                            ylabel="Prompt Style", title=DATASET_LABELS[ds])
    fig.suptitle("Primary Topic Directional Bias by Dataset and Prompt Style\n"
                 "(Averaged across Models, ±SD shown in annotations)",
                 fontweight="bold", fontsize=18, y=1.04)
    plt.tight_layout()
    fig.savefig(OUT / "08b_topic_bias_by_dataset_prompt.png", bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 08b_topic_bias_by_dataset_prompt.png")

    # 08c: 3×3 grid (datasets × models), panel rows=prompts
    all_pivs = []
    for ds in DATASET_ORDER:
        for prov in PROVIDER_ORDER:
            sub = pt[(pt["dataset"] == ds) & (pt["provider"] == prov)]
            piv = sub.pivot_table(index="prompt_style", columns="category",
                                  values="directional_bias", aggfunc="mean")
            all_pivs.append(piv.reindex(index=PROMPT_ORDER, columns=top_topics))

    vmax = _shared_vmax(all_pivs)
    fig, axes = plt.subplots(3, 3, figsize=(19, 14))
    for di, ds in enumerate(DATASET_ORDER):
        for pi, prov in enumerate(PROVIDER_ORDER):
            ax  = axes[di][pi]
            sub = pt[(pt["dataset"] == ds) & (pt["provider"] == prov)]
            piv = sub.pivot_table(index="prompt_style", columns="category",
                                  values="directional_bias", aggfunc="mean")
            piv = piv.reindex(index=PROMPT_ORDER, columns=top_topics)
            _draw_topic_heatmap(
                ax, piv, _make_topic_annot(piv), vmax,
                col_lbls=_topic_col_labels(pt, top_topics, ds, prov=prov),
                row_lbls=[PROMPT_LABELS[p] for p in PROMPT_ORDER],
                show_cbar=(pi == 2), show_ylabel=(pi == 0),
                ylabel=DATASET_LABELS[ds],
                title=PROVIDER_LABELS[prov] if di == 0 else "",
                annot_fs=8)
    fig.suptitle("Primary Topic Directional Bias by Dataset, Model, and Prompt Style",
                 fontweight="bold", fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(OUT / "08c_topic_bias_by_dataset_model_prompt.png",
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  ✓ 08c_topic_bias_by_dataset_model_prompt.png")

# ============================================================================
# POOL DISTRIBUTIONS CSV
# ============================================================================

def save_pool_distributions():
    print("\n" + "="*70)
    print("SAVING pool_distributions.csv")
    print("="*70)

    df   = pd.read_csv(DIR_BIAS_CSV)
    rows = []

    cat = df[df["feature_type"] == "categorical"]
    for (feature, category, dataset), grp in cat.groupby(["feature", "category", "dataset"]):
        rows.append({
            "feature": feature, "dataset": dataset, "feature_type": "categorical",
            "category": category,
            "pool_proportion":     round(grp["prop_pool"].mean(), 6),
            "pool_proportion_std": round(grp["prop_pool"].std(),  6),
            "pool_mean": np.nan, "pool_mean_std": np.nan,
        })

    cont = df[df["feature_type"] == "continuous"]
    for (feature, dataset), grp in cont.groupby(["feature", "dataset"]):
        rows.append({
            "feature": feature, "dataset": dataset, "feature_type": "continuous",
            "category": np.nan,
            "pool_proportion": np.nan, "pool_proportion_std": np.nan,
            "pool_mean":     round(grp["mean_pool"].mean(), 6),
            "pool_mean_std": round(grp["mean_pool"].std(),  6),
        })

    pd.DataFrame(rows).sort_values(
        ["feature_type", "feature", "dataset", "category"]
    ).to_csv(OUT / "pool_distributions.csv", index=False)
    print("  ✓ pool_distributions.csv")

# ============================================================================
# FIGURE 10: DEMOGRAPHIC BIAS BY MODEL (one figure per demographic variable)
#   3 subplots per figure (one per provider), rows = prompt styles, cols = categories
# ============================================================================

_DEMO_FEATURES = {
    "author_political_leaning": {
        "display_name":    "Author Political Leaning",
        "categories":      ["left", "center-left", "center", "center-right", "right", "unknown"],
        "category_labels": ["Left", "Ctr-Left", "Center", "Ctr-Right", "Right", "Unknown"],
    },
    "author_gender": {
        "display_name":    "Author Gender",
        "categories":      ["female", "male", "non-binary", "unknown"],
        "category_labels": ["Female", "Male", "Non-Binary", "Unknown"],
    },
    "author_is_minority": {
        "display_name":    "Author Minority Status",
        "categories":      ["yes", "no", "unknown"],
        "category_labels": ["Minority", "Non-Minority", "Unknown"],
    },
}


def _load_demo_bias():
    """Load and zero-sum-normalise demographic directional bias (Twitter/X only)."""
    dir_bias = pd.read_csv(DIR_BIAS_CSV)
    dir_bias = dir_bias[
        ~dir_bias["feature"].isin(AUTHOR_FEATURES) |
        dir_bias["dataset"].isin(DEMOGRAPHIC_DATASETS)
    ]
    categorical_features = list(_DEMO_FEATURES.keys())
    normalized_rows = []
    for feature in categorical_features:
        fdata = dir_bias[dir_bias["feature"] == feature].copy()
        for (ds, prov, prompt), grp in fdata.groupby(["dataset", "provider", "prompt_style"]):
            bias_sum   = grp["directional_bias"].sum()
            correction = bias_sum / len(grp) if abs(bias_sum) > 1e-10 else 0
            grp = grp.copy()
            grp["directional_bias"] -= correction
            normalized_rows.append(grp)
    dir_bias = dir_bias[~dir_bias["feature"].isin(categorical_features)]
    return pd.concat([dir_bias] + normalized_rows, ignore_index=True)


def plot_10_demographic_by_model():
    print("\n" + "="*70)
    print("FIGURE 10 — Demographic bias by model (Twitter/X only)")
    print("="*70)

    if not WITH_DEMOGRAPHICS:
        print("  skipped (WITH_DEMOGRAPHICS = False)")
        return

    dir_bias = _load_demo_bias()

    for feature, finfo in _DEMO_FEATURES.items():
        fdata = dir_bias[(dir_bias["feature"] == feature) & (dir_bias["dataset"] == "twitter")]
        if len(fdata) == 0:
            print(f"  skipped {feature} (no data)")
            continue

        avail      = [c for c in finfo["categories"] if c in fdata["category"].unique()]
        cat_lookup = dict(zip(finfo["categories"], finfo["category_labels"]))
        col_labels = [cat_lookup.get(c, c) for c in avail]

        # Shared colour scale across all three models
        all_vals = fdata["directional_bias"].dropna()
        max_abs  = max(abs(all_vals.min()), abs(all_vals.max())) if len(all_vals) else 1.0

        fig, axes = plt.subplots(1, len(PROVIDER_ORDER), figsize=(20, 8.05))

        for idx, provider in enumerate(PROVIDER_ORDER):
            ax       = axes[idx]
            prov_data = fdata[fdata["provider"] == provider]

            mean_agg = prov_data.groupby(["prompt_style", "category"])["directional_bias"].mean().reset_index()
            std_agg  = prov_data.groupby(["prompt_style", "category"])["directional_bias"].std().reset_index()

            piv_m = mean_agg.pivot(index="prompt_style", columns="category", values="directional_bias")
            piv_s = std_agg.pivot( index="prompt_style", columns="category", values="directional_bias")

            piv_m = piv_m.reindex(index=PROMPT_ORDER, columns=avail)
            piv_s = piv_s.reindex(index=PROMPT_ORDER, columns=avail)

            # Std for average row/col
            std_col_vals = piv_m.std(axis=1).values   # per prompt style, across categories
            std_row_vals = piv_m.std(axis=0).values   # per category, across prompt styles
            overall_std  = float(np.nanstd(piv_m.values))

            avg_col            = piv_m.mean(axis=1)
            avg_row            = pd.Series(piv_m.mean(axis=0), name="Average")
            piv_m              = pd.concat([piv_m, avg_row.to_frame().T])
            avg_col["Average"] = avg_col.mean()
            piv_m["Average"]   = avg_col

            piv_m.columns = col_labels + ["Average"]
            piv_m.index   = [PROMPT_LABELS.get(p, p) for p in piv_m.index]

            n_rows, n_cols = piv_m.shape
            annot = np.empty_like(piv_m, dtype=object)
            for i in range(n_rows):
                for j in range(n_cols):
                    val = piv_m.iloc[i, j]
                    if pd.isna(val):
                        annot[i, j] = ""
                    elif i == n_rows - 1 and j == n_cols - 1:
                        annot[i, j] = f"{val:.3f}\n±{overall_std:.3f}"
                    elif i == n_rows - 1:
                        annot[i, j] = f"{val:.3f}\n±{std_row_vals[j]:.3f}"
                    elif j == n_cols - 1:
                        annot[i, j] = f"{val:.3f}\n±{std_col_vals[i]:.3f}"
                    else:
                        annot[i, j] = f"{val:.3f}"

            show_cbar = (idx == len(PROVIDER_ORDER) - 1)
            cbar_label = "Directional Bias\n← Under | Over-represented →"
            sns.heatmap(piv_m, annot=annot, fmt="", cmap=CMAP_DIVG,
                        center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                        cbar=show_cbar,
                        cbar_kws={"label": cbar_label} if show_cbar else {},
                        linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 15})

            if show_cbar and ax.collections:
                cbar = ax.collections[0].colorbar
                if cbar:
                    cbar.ax.tick_params(labelsize=16)
                    cbar.set_label(cbar_label, fontsize=17, fontweight="bold")

            ax.axhline(y=n_rows - 1, color="black", linewidth=2.5)
            ax.axvline(x=n_cols - 1, color="black", linewidth=2.5)
            ax.set_title(PROVIDER_LABELS[provider], fontweight="bold", fontsize=20, pad=10)
            ax.set_xlabel("Category", fontsize=18, fontweight="bold")
            ax.set_ylabel("Prompt Style" if idx == 0 else "", fontsize=18, fontweight="bold")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=15)
            if idx > 0:
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=15)

        fig.suptitle(
            f"{finfo['display_name']} — Directional Bias by Model and Prompt Style (Twitter/X)",
            fontweight="bold", fontsize=20, y=1.02)
        plt.tight_layout(w_pad=3)

        slug  = feature.replace("author_", "")
        fname = f"10_demo_bias_{slug}_by_model.png"
        fig.savefig(OUT / fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  ✓ {fname}")

        # Save CSV
        mean_df = fdata.groupby(["provider", "prompt_style", "category"])["directional_bias"].mean().reset_index()
        std_df  = fdata.groupby(["provider", "prompt_style", "category"])["directional_bias"].std().reset_index()
        mean_df = mean_df.rename(columns={"directional_bias": "mean_bias"})
        std_df  = std_df.rename(columns={"directional_bias": "std_bias"})
        csv_df  = mean_df.merge(std_df, on=["provider", "prompt_style", "category"])
        csv_df["provider_label"] = csv_df["provider"].map(PROVIDER_LABELS)
        csv_fname = f"10_demo_bias_{slug}_by_model.csv"
        csv_df.to_csv(OUT / csv_fname, index=False)
        print(f"  ✓ {csv_fname}")


# ============================================================================
# FIGURE 09: RAW DIRECTIONAL BIAS BY DATASET (one figure per metric)
# ============================================================================

_METRICS_09 = {
    "avg_word_length": {
        "title":  "Average Word Length Directional Bias by Dataset, Model, and Prompt Style",
        "ylabel": "Directional Bias (chars/word)\n← Shorter | Longer →",
    },
    "polarization_score": {
        "title":  "Polarization Directional Bias by Dataset, Model, and Prompt Style",
        "ylabel": "Directional Bias\n← Less | More Polarized →",
    },
    "sentiment_polarity": {
        "title":  "Sentiment Polarity Directional Bias by Dataset, Model, and Prompt Style",
        "ylabel": "Directional Bias\n← More Negative | More Positive →",
    },
    "toxicity": {
        "title":  "Toxicity Directional Bias by Dataset, Model, and Prompt Style",
        "ylabel": "Directional Bias\n← Less | More Toxic →",
    },
}


def _plot_09_single_metric(feature, minfo):
    data = _load_metric_bias(feature)

    # Shared colour scale across all three datasets
    all_vals = data["directional_bias"].dropna()
    max_abs  = max(abs(all_vals.min()), abs(all_vals.max())) if len(all_vals) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 8.05))

    for idx, ds in enumerate(DATASET_ORDER):
        ax      = axes[idx]
        ds_data = data[data["dataset"] == ds]

        mean_agg = ds_data.groupby(["provider", "prompt_style"])["directional_bias"].mean().reset_index()
        std_agg  = ds_data.groupby(["provider", "prompt_style"])["directional_bias"].std().reset_index()

        piv_m = mean_agg.pivot(index="prompt_style", columns="provider", values="directional_bias")
        piv_s = std_agg.pivot( index="prompt_style", columns="provider", values="directional_bias")

        piv_m = piv_m.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)
        piv_s = piv_s.reindex(index=PROMPT_ORDER, columns=PROVIDER_ORDER)

        # Std for summary row/col (spread across models / prompt styles)
        std_col_vals = piv_m.std(axis=1).values   # per prompt style, across models
        std_row_vals = piv_m.std(axis=0).values   # per model, across prompt styles
        overall_std  = float(np.nanstd(piv_m.values))

        avg_col             = piv_m.mean(axis=1)
        avg_row             = pd.Series(piv_m.mean(axis=0), name="Average")
        piv_m               = pd.concat([piv_m, avg_row.to_frame().T])
        avg_col["Average"]  = avg_col.mean()
        piv_m["Average"]    = avg_col

        short           = {p: PROVIDER_LABELS[p] for p in PROVIDER_ORDER}
        short["Average"] = "Average"
        piv_m.columns   = [short.get(p, p) for p in piv_m.columns]
        piv_m.index     = [PROMPT_LABELS.get(p, p) for p in piv_m.index]

        n_rows, n_cols = piv_m.shape
        annot = np.empty_like(piv_m, dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                val = piv_m.iloc[i, j]
                if pd.isna(val):
                    annot[i, j] = ""
                elif i == n_rows - 1 and j == n_cols - 1:
                    annot[i, j] = f"{val:.3f}\n±{overall_std:.3f}"
                elif i == n_rows - 1:
                    annot[i, j] = f"{val:.3f}\n±{std_row_vals[j]:.3f}"
                elif j == n_cols - 1:
                    annot[i, j] = f"{val:.3f}\n±{std_col_vals[i]:.3f}"
                else:
                    annot[i, j] = f"{val:.3f}"

        show_cbar = (idx == len(DATASET_ORDER) - 1)
        sns.heatmap(piv_m, annot=annot, fmt="", cmap=CMAP_DIVG,
                    center=0, vmin=-max_abs, vmax=max_abs, ax=ax,
                    cbar=show_cbar,
                    cbar_kws={"label": minfo["ylabel"]} if show_cbar else {},
                    linewidths=0.5, linecolor="gray", annot_kws={"fontsize": 15})

        if show_cbar and ax.collections:
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.tick_params(labelsize=16)
                cbar.set_label(minfo["ylabel"], fontsize=17, fontweight="bold")

        ax.axhline(y=n_rows - 1, color="black", linewidth=2.5)
        ax.axvline(x=n_cols - 1, color="black", linewidth=2.5)
        ax.set_title(DATASET_LABELS[ds], fontweight="bold", fontsize=20, pad=10)
        ax.set_xlabel("Model", fontsize=18, fontweight="bold")
        ax.set_ylabel("Prompt Style" if idx == 0 else "", fontsize=18, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=15)
        if idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center", fontsize=15)

    fig.suptitle(minfo["title"], fontweight="bold", fontsize=20, y=1.02)
    plt.tight_layout(w_pad=3)
    return fig


def plot_09_raw_bias_heatmaps():
    print("\n" + "="*70)
    print("FIGURE 09 — Raw directional bias by dataset (one figure per metric)")
    print("="*70)

    labels = {
        "avg_word_length":   "09a",
        "polarization_score":"09b",
        "sentiment_polarity":"09c",
        "toxicity":          "09d",
    }
    for feature, minfo in _METRICS_09.items():
        fig = _plot_09_single_metric(feature, minfo)
        tag  = labels[feature]
        fname = f"{tag}_raw_bias_{feature}_by_dataset.png"
        fig.savefig(OUT / fname, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"  ✓ {fname}")

        # Save CSV
        data = _load_metric_bias(feature)
        mean_df = data.groupby(["dataset", "provider", "prompt_style"])["directional_bias"].mean().reset_index()
        std_df  = data.groupby(["dataset", "provider", "prompt_style"])["directional_bias"].std().reset_index()
        mean_df = mean_df.rename(columns={"directional_bias": "mean_bias"})
        std_df  = std_df.rename(columns={"directional_bias": "std_bias"})
        csv_df  = mean_df.merge(std_df, on=["dataset", "provider", "prompt_style"])
        csv_df["provider_label"] = csv_df["provider"].map(PROVIDER_LABELS)
        csv_fname = f"{tag}_raw_bias_{feature}_by_dataset.csv"
        csv_df.to_csv(OUT / csv_fname, index=False)
        print(f"  ✓ {csv_fname}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print(f"GENERATING ALL PAPER PLOTS  (WITH_DEMOGRAPHICS = {WITH_DEMOGRAPHICS})")
    print(f"Output → {OUT}")
    print("=" * 70)

    missing = [p for p in [SUMMARY_CSV, DIR_BIAS_CSV, IMPORTANCE_CSV] if not p.exists()]
    if WITH_DEMOGRAPHICS and not INFERRED_BIO_CSV.exists():
        missing.append(INFERRED_BIO_CSV)
    if missing:
        print("\nERROR — missing input files:")
        for p in missing:
            print(f"  {p}")
        return

    comp_df = load_summary(WITH_DEMOGRAPHICS)
    print(f"\nLoaded {len(comp_df)} rows from pool_vs_recommended_summary.csv")

    save_pool_distributions()
    plot_01_aggregated_bar(comp_df)
    plot_02_bias_by_prompt(comp_df)
    plot_03_normalized_bias(comp_df)
    if WITH_DEMOGRAPHICS:
        plot_04_demographics()
        plot_10_demographic_by_model()
    plot_05_content_safety_heatmap()
    plot_06_content_safety_bars()
    plot_07_feature_importance(WITH_DEMOGRAPHICS)
    plot_08_topic_heatmaps()
    plot_09_raw_bias_heatmaps()

    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name:<65} {f.stat().st_size // 1024:>4} KB")


if __name__ == "__main__":
    main()
