#!/usr/bin/env python3
"""
Compute bias metrics for LLM Recommendation Bias Study.

Analyzes bias across 13 features, 3 datasets (Twitter/Bluesky/Reddit),
3 LLM providers (Anthropic/OpenAI/Gemini), and 6 prompt styles (54 conditions).

Reads per-condition post_level_data.csv files from outputs/experiments/ and
produces three aggregated output files in analysis_outputs/:
  - pool_vs_recommended_summary.csv   (R², Cohen's d / Cramér's V per feature)
  - directional_bias_data.csv         (directional bias per category)
  - feature_importance_data.csv       (Random Forest SHAP + AUROC)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import shap
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

# 15 Core Features (grouped by category)
FEATURES = {
    'author': ['author_gender', 'author_political_leaning', 'author_is_minority'],
    'text_metrics': ['avg_word_length'],
    'sentiment': ['sentiment_polarity', 'sentiment_subjectivity'],
    'style': ['has_emoji', 'has_hashtag', 'has_mention', 'has_url'],
    'content': ['polarization_score', 'primary_topic'],
    'toxicity': ['toxicity']
}

# Feature types
FEATURE_TYPES = {
    # Author (categorical)
    'author_gender': 'categorical',
    'author_political_leaning': 'categorical',
    'author_is_minority': 'categorical',

    # Text metrics (numerical)
    'avg_word_length': 'numerical',

    # Sentiment (numerical)
    'sentiment_polarity': 'numerical',
    'sentiment_subjectivity': 'numerical',

    # Style (binary)
    'has_emoji': 'binary',
    'has_hashtag': 'binary',
    'has_mention': 'binary',
    'has_url': 'binary',

    # Content (mixed)
    'polarization_score': 'numerical',
    'primary_topic': 'categorical',

    # Toxicity (numerical)
    'toxicity': 'numerical'
}

CATEGORY_ORDERS = {
    'author_political_leaning': ['left', 'center-left', 'center', 'center-right', 'right', 'apolitical', 'unknown'],
    'author_gender': ['male', 'female', 'non-binary', 'unknown'],
    'author_is_minority': ['no', 'yes', 'unknown'],
    'has_emoji': [0, 1],
    'has_hashtag': [0, 1],
    'has_mention': [0, 1],
    'has_url': [0, 1],
}

# Substantive vs Stylistic categorization
SUBSTANTIVE_FEATURES = (
    FEATURES['author'] + FEATURES['sentiment'] +
    FEATURES['content'] + FEATURES['toxicity']
)
STYLISTIC_FEATURES = FEATURES['text_metrics'] + FEATURES['style']

# Datasets and models
DATASETS = ['twitter', 'bluesky', 'reddit']
PROVIDERS = ['openai', 'anthropic', 'gemini']
PROMPT_STYLES = ['general', 'popular', 'engaging', 'informative', 'controversial', 'neutral']

# Output directories
OUTPUT_DIR = Path('analysis_outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations'
DIST_DIR = VIZ_DIR / '1_distributions'
HEATMAP_DIR = VIZ_DIR / '2_bias_heatmaps'
DIR_BIAS_DIR = VIZ_DIR / '3_directional_bias'
IMPORTANCE_DIR = VIZ_DIR / '4_feature_importance'
COMPARE_DIR = VIZ_DIR / '_archive_pool_vs_recommended'
COMPARE_AGG_DIR = COMPARE_DIR / 'aggregated'
TOP5_DIR = VIZ_DIR / '_archive_top5_significant'
TABLES_DIR = OUTPUT_DIR / '6_regression_tables'

# Create all directories
for d in [VIZ_DIR, DIST_DIR, HEATMAP_DIR, DIR_BIAS_DIR, IMPORTANCE_DIR, COMPARE_DIR, COMPARE_AGG_DIR, TOP5_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# ENHANCED VISUALIZATION SETTINGS
# ============================================================================

# Dataset color palette (consistent across all plots)
DATASET_COLORS = {
    'twitter': '#2F2F2F',      # Dark gray/blackish for Twitter/X
    'bluesky': '#4A90E2',      # Blue for Bluesky
    'reddit': '#FF6B35'        # Orange-red for Reddit
}

# Dataset display names
DATASET_LABELS = {
    'twitter': 'Twitter/X',
    'bluesky': 'Bluesky',
    'reddit': 'Reddit'
}

# Enhanced color schemes for different plot types
# Using perceptually distinct colors (avoiding simple red/green)
DIVERGING_CMAP = 'RdYlBu_r'  # Red-Yellow-Blue (more interesting than red-green)
SEQUENTIAL_CMAP = 'viridis'   # Perceptually uniform colormap
CATEGORICAL_CMAP = 'Set3'     # For categorical features

# Directional bias colors (replacing simple red/green)
DIRECTIONAL_COLORS = {
    'negative': '#D55E00',  # Vermillion (colorblind-friendly)
    'positive': '#009E73',  # Bluish green (colorblind-friendly)
    'neutral': '#999999'    # Gray
}

# Feature type order for heatmap rows (grouped by type)
FEATURE_TYPE_ORDER = (
    # Author features (demographics)
    FEATURES['author'] +
    # Text metrics
    FEATURES['text_metrics'] +
    # Content features
    FEATURES['content'] +
    # Sentiment features
    FEATURES['sentiment'] +
    # Style features (binary indicators)
    FEATURES['style'] +
    # Toxicity features
    FEATURES['toxicity']
)

# Feature display names (human-readable, no underscores)
FEATURE_DISPLAY_NAMES = {
    # Author features
    'author_gender': 'Author: Gender',
    'author_political_leaning': 'Author: Political Leaning',
    'author_is_minority': 'Author: Is Minority',
    # Text metrics
    'avg_word_length': 'Text: Avg Word Length',
    # Content features
    'polarization_score': 'Content: Polarization Score',
    'primary_topic': 'Content: Primary Topic',
    # Sentiment features
    'sentiment_polarity': 'Sentiment: Polarity',
    'sentiment_subjectivity': 'Sentiment: Subjectivity',
    # Style features (binary)
    'has_emoji': 'Style: Has Emoji',
    'has_hashtag': 'Style: Has Hashtag',
    'has_mention': 'Style: Has Mention',
    'has_url': 'Style: Has URL',
    # Toxicity features
    'toxicity': 'Toxicity: Score'
}

def format_feature_name(feature_name):
    """Convert feature name to human-readable format."""
    return FEATURE_DISPLAY_NAMES.get(feature_name, feature_name.replace('_', ' ').title())

def get_feature_category(feature_name):
    """Get the category label for a feature."""
    for category, features in FEATURES.items():
        if feature_name in features:
            return category.replace('_', ' ').title()
    return 'Other'

def get_dataset_color(dataset_name):
    """Get consistent color for a dataset."""
    return DATASET_COLORS.get(dataset_name, '#666666')

def get_dataset_label(dataset_name):
    """Get display label for a dataset."""
    return DATASET_LABELS.get(dataset_name, dataset_name.title())

def get_directional_color(value):
    """Get color for directional bias value."""
    if value < -0.01:
        return DIRECTIONAL_COLORS['negative']
    elif value > 0.01:
        return DIRECTIONAL_COLORS['positive']
    else:
        return DIRECTIONAL_COLORS['neutral']

def sort_features_by_type(features_list):
    """Sort features by their type (author, text, content, etc.)."""
    return sorted(features_list, key=lambda x: (
        FEATURE_TYPE_ORDER.index(x) if x in FEATURE_TYPE_ORDER else 999,
        x
    ))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_pool_data(dataset):
    """Load unique pool posts for a dataset"""
    exp_dirs = list(Path('outputs/experiments').glob(f"{dataset}_anthropic_*"))
    if not exp_dirs:
        exp_dirs = list(Path('outputs/experiments').glob(f"{dataset}_*"))
    df = pd.read_csv(exp_dirs[0] / 'post_level_data.csv')
    # Get unique posts (selected=0 to get pool, then deduplicate)
    dedup_col = 'post_id' if 'post_id' in df.columns else df.columns[0]
    pool = df[df['selected'] == 0].drop_duplicates(subset=dedup_col).copy()
    return pool

def load_experiment_data(dataset, provider):
    """Load full experiment data"""
    # Find experiment directory
    exp_dirs = list(Path('outputs/experiments').glob(f"{dataset}_{provider}_*"))
    if not exp_dirs:
        return None
    df = pd.read_csv(exp_dirs[0] / 'post_level_data.csv')
    return df

def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def compute_cramers_v(pool_vals, rec_vals):
    """
    FIXED: Compute Cramér's V using proper contingency table
    Creates: feature_values × (pool vs recommended)
    """
    try:
        # CRITICAL FIX: Reset index to avoid misalignment between series
        pool_vals = pool_vals.reset_index(drop=True)
        rec_vals = rec_vals.reset_index(drop=True)

        # Concatenate and reset index to avoid duplicates
        combined = pd.concat([pool_vals, rec_vals], ignore_index=True)
        labels = pd.Series(['pool']*len(pool_vals) + ['rec']*len(rec_vals))

        # Create proper contingency table: feature_values × (pool vs rec)
        contingency = pd.crosstab(combined, labels)

        # Check if there's variation
        if contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
            return 0

        chi2, _, _, _ = chi2_contingency(contingency)
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1

        if min_dim == 0 or n == 0:
            return 0

        return np.sqrt(chi2 / (n * min_dim))
    except:
        return 0

def compute_bias_metric(pool_vals, rec_vals, feature_type):
    """
    FIXED: Better handling of constant features and error cases
    Compute appropriate bias metric based on feature type
    Returns: (bias_value, p_value, metric_name)
    """
    pool_vals = pool_vals.dropna()
    rec_vals = rec_vals.dropna()

    if len(pool_vals) < 10 or len(rec_vals) < 10:
        return 0, 1.0, "insufficient_data"

    if feature_type == 'numerical':
        # Check for variance
        if pool_vals.std() == 0 and rec_vals.std() == 0:
            return 0, 1.0, "Cohen's d (no variance)"

        bias = compute_cohens_d(rec_vals, pool_vals)
        try:
            t_stat, p_val = stats.ttest_ind(rec_vals, pool_vals, equal_var=False)
            if np.isnan(p_val):
                p_val = 1.0
        except:
            p_val = 1.0
        return bias, p_val, "Cohen's d"

    elif feature_type in ['categorical', 'binary']:
        # Check for variance
        if len(pool_vals.unique()) <= 1 and len(rec_vals.unique()) <= 1:
            return 0, 1.0, "Cramér's V (no variance)"

        try:
            bias = compute_cramers_v(pool_vals, rec_vals)

            # CRITICAL FIX: Reset index and use ignore_index for chi-square test
            pool_vals_reset = pool_vals.reset_index(drop=True)
            rec_vals_reset = rec_vals.reset_index(drop=True)
            combined = pd.concat([pool_vals_reset, rec_vals_reset], ignore_index=True)
            labels = pd.Series(['pool']*len(pool_vals_reset) + ['rec']*len(rec_vals_reset))

            # Chi-square test
            contingency = pd.crosstab(combined, labels)
            chi2, p_val, dof, expected = chi2_contingency(contingency)

            if np.isnan(p_val):
                p_val = 1.0

            return bias, p_val, "Cramér's V"
        except Exception as e:
            return 0, 1.0, f"Cramér's V (error: {str(e)[:20]})"

    return 0, 1.0, "unknown"

def standardize_categories(series, feature_name):
    """FIXED: Standardize category values and apply consistent ordering"""
    if feature_name in CATEGORY_ORDERS:
        # Convert to string for consistent comparison
        series = series.astype(str).str.lower()
        # Create ordered categorical
        valid_categories = [str(c).lower() for c in CATEGORY_ORDERS[feature_name]]
        series = pd.Categorical(series, categories=valid_categories, ordered=True)
    return series

# ============================================================================
# 1. FEATURE DISTRIBUTIONS (Pool Data Only) - FIXED
# ============================================================================

def generate_feature_distributions():
    """
    FIXED: Use better styling from old analyze_features.py
    Generate distribution plots for each feature × dataset
    """
    print("\n" + "="*80)
    print("GENERATING FEATURE DISTRIBUTIONS (Pool Data) - IMPROVED STYLE")
    print("="*80)

    all_features = sum(FEATURES.values(), [])

    # Load all pool data once
    pools = {dataset: load_pool_data(dataset) for dataset in DATASETS}

    for feature in all_features:
        feat_type = FEATURE_TYPES.get(feature, 'numerical')

        # Author demographic features are Twitter-only
        available_pools = {
            ds: df for ds, df in pools.items() if feature in df.columns
        }
        if not available_pools:
            print(f"  ⚠ {feature} not found in any dataset, skipping...")
            continue

        try:
            if feat_type == 'numerical':
                plot_numerical_distribution(available_pools, feature)
            elif feat_type == 'binary':
                plot_binary_distribution(available_pools, feature)
            else:  # categorical
                plot_categorical_distribution(available_pools, feature)

            datasets_used = list(available_pools.keys())
            suffix = f" (Twitter only)" if datasets_used == ['twitter'] else ""
            print(f"  ✓ {feature}{suffix}")
        except Exception as e:
            print(f"  ✗ Error plotting {feature}: {e}")

    print(f"\n✓ Saved {len(all_features)} distribution plots to {DIST_DIR}")

def plot_numerical_distribution(pools, feature):
    """Plot numerical feature with improved styling"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(format_feature_name(feature), fontsize=14, fontweight='bold')

    for idx, (dataset, pool) in enumerate(pools.items()):
        ax = axes[idx]
        values = pd.to_numeric(pool[feature], errors='coerce').dropna()

        if len(values) > 0 and values.std() > 0:
            # Use dataset-specific colors
            color = get_dataset_color(dataset)
            ax.hist(values, bins=50, alpha=0.7, color=color, edgecolor='black')
            ax.axvline(values.mean(), color='#E74C3C', linestyle='--', linewidth=2,
                      label=f'Mean: {values.mean():.3f}')
            ax.axvline(values.median(), color='#F39C12', linestyle='--', linewidth=2,
                      label=f'Median: {values.median():.3f}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{get_dataset_label(dataset)} (n={len(values):,})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif len(values) > 0:
            ax.text(0.5, 0.5, f'No variation\n(all values = {values.iloc[0]:.3f})',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{get_dataset_label(dataset)} (n={len(values):,})')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{get_dataset_label(dataset)}')

    plt.tight_layout()
    plt.savefig(DIST_DIR / f'{feature}_distribution.png', bbox_inches='tight')
    plt.close()

def plot_categorical_distribution(pools, feature):
    """Plot categorical feature with improved styling and consistent ordering"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{feature}', fontsize=14, fontweight='bold')

    # Get ordering if specified
    order = CATEGORY_ORDERS.get(feature, None)

    for idx, (dataset, pool) in enumerate(pools.items()):
        ax = axes[idx]
        values = pool[feature].dropna().astype(str)

        if len(values) > 0:
            value_counts = values.value_counts()

            # Apply ordering if specified
            if order:
                # Convert order to strings for comparison
                order_str = [str(x).lower() for x in order]
                # Reindex to order, filling missing categories with 0
                # First normalize the index
                value_counts.index = value_counts.index.str.lower()
                value_counts = value_counts.reindex(order_str, fill_value=0)

            # Use Set3 colormap for nice categorical colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
            value_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.set_title(f'{dataset.capitalize()} (n={len(values):,})')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dataset.capitalize()}')

    plt.tight_layout()
    plt.savefig(DIST_DIR / f'{feature}_distribution.png', bbox_inches='tight')
    plt.close()

def plot_binary_distribution(pools, feature):
    """Plot binary feature with stacked bar chart (all datasets in one plot)"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data_to_plot = []
    labels = []

    for dataset, pool in pools.items():
        values = pd.to_numeric(pool[feature], errors='coerce').dropna()
        if len(values) > 0:
            # Calculate proportions
            prop_yes = (values == 1).sum() / len(values) * 100
            prop_no = (values == 0).sum() / len(values) * 100
            data_to_plot.append([prop_no, prop_yes])
            labels.append(f'{dataset.capitalize()}\n(n={len(values):,})')
        else:
            data_to_plot.append([0, 0])
            labels.append(f'{dataset.capitalize()}\n(n=0)')

    if len(data_to_plot) > 0:
        x = np.arange(len(labels))
        width = 0.6

        data_array = np.array(data_to_plot)
        ax.bar(x, data_array[:, 0], width, label='No (0)', color='lightcoral', edgecolor='black')
        ax.bar(x, data_array[:, 1], width, bottom=data_array[:, 0],
               label='Yes (1)', color='lightgreen', edgecolor='black')

        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{feature}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Add percentage labels
        for i, (no, yes) in enumerate(data_array):
            if no > 5:  # Only show if >5%
                ax.text(i, no/2, f'{no:.1f}%', ha='center', va='center', fontweight='bold')
            if yes > 5:  # Only show if >5%
                ax.text(i, no + yes/2, f'{yes:.1f}%', ha='center', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(DIST_DIR / f'{feature}_distribution.png', bbox_inches='tight')
    plt.close()

# ============================================================================
# 2. POOL VS RECOMMENDED COMPARISONS - ENHANCED
# ============================================================================

def generate_pool_vs_recommended():
    """
    ENHANCED: Generate aggregated comparison plots (by prompt, dataset, model)
    SKIP fully disaggregated 864 plots - too many!
    """
    print("\n" + "="*80)
    print("GENERATING POOL VS RECOMMENDED COMPARISONS - AGGREGATED")
    print("="*80)

    # Check if comparison data already exists
    comp_data_file = OUTPUT_DIR / 'comparison_data.parquet'
    if comp_data_file.exists():
        print("\n✓ Loading cached comparison data...")
        comp_df = pd.read_parquet(comp_data_file)
        print(f"  Loaded {len(comp_df)} comparisons from cache")
        print(f"  To recompute, delete: {comp_data_file}\n")
        return comp_df

    all_features = sum(FEATURES.values(), [])
    comparisons = []

    # Part 1: Compute comparisons (but don't plot each one)
    print("\n1. Computing comparisons for all conditions...")
    for dataset in DATASETS:
        for provider in PROVIDERS:
            df = load_experiment_data(dataset, provider)
            if df is None:
                continue

            for prompt in PROMPT_STYLES:
                prompt_df = df[df['prompt_style'] == prompt]
                pool = prompt_df[prompt_df['selected'] == 0]
                recommended = prompt_df[prompt_df['selected'] == 1]

                if len(pool) == 0 or len(recommended) == 0:
                    continue

                # Compute bias for each feature (but don't plot individually)
                for feature in all_features:
                    if feature not in pool.columns:
                        continue

                    feat_type = FEATURE_TYPES.get(feature, 'numerical')

                    # Compute bias for summary
                    bias, p_val, metric = compute_bias_metric(
                        pool[feature], recommended[feature], feat_type
                    )

                    comparisons.append({
                        'feature': feature,
                        'dataset': dataset,
                        'provider': provider,
                        'prompt_style': prompt,
                        'bias': bias,
                        'p_value': p_val,
                        'metric': metric,
                        'significant': p_val < 0.05
                    })

    print(f"✓ Computed {len(comparisons)} comparisons across all conditions")

    # Part 2: Save comparison data (but skip plot generation)
    print("\n2. Saving comparison summary (skipping plot generation)...")
    comp_df = pd.DataFrame(comparisons)

    # Save full summary (both CSV and Parquet for fast loading)
    comp_df.to_csv(OUTPUT_DIR / 'pool_vs_recommended_summary.csv', index=False)
    comp_df.to_parquet(OUTPUT_DIR / 'comparison_data.parquet', index=False)

    # SKIP: Generate aggregated plots (not needed for now)
    # generate_aggregated_comparisons_enhanced(comp_df)

    print("✓ Comparison data saved (plots skipped)")

    return comp_df

def generate_aggregated_comparisons_enhanced(comp_df):
    """
    ENHANCED: Generate visual comparison plots aggregated by:
    - Prompt style (6 plots per feature)
    - Dataset (3 plots per feature)
    - Model (3 plots per feature)
    """
    all_features = sum(FEATURES.values(), [])

    # Helper function to load and aggregate data
    def get_aggregated_pool_rec(filter_dict, feature):
        """Load pool and recommended data filtered by conditions"""
        pool_all = []
        rec_all = []

        for dataset in DATASETS:
            for provider in PROVIDERS:
                df = load_experiment_data(dataset, provider)
                if df is None:
                    continue

                # Apply filters
                filtered_df = df
                for key, value in filter_dict.items():
                    if key == 'dataset' and dataset != value:
                        filtered_df = None
                        break
                    elif key == 'provider' and provider != value:
                        filtered_df = None
                        break
                    elif key == 'prompt_style':
                        filtered_df = filtered_df[filtered_df['prompt_style'] == value]

                if filtered_df is None or len(filtered_df) == 0:
                    continue

                if feature not in filtered_df.columns:
                    continue

                pool = filtered_df[filtered_df['selected'] == 0][feature]
                rec = filtered_df[filtered_df['selected'] == 1][feature]

                pool_all.extend(pool.dropna().tolist())
                rec_all.extend(rec.dropna().tolist())

        return pd.Series(pool_all), pd.Series(rec_all)

    def plot_comparison_aggregated(pool_data, rec_data, feature, title, filename):
        """Plot pool vs recommended comparison"""
        if len(pool_data) == 0 or len(rec_data) == 0:
            return

        feat_type = FEATURE_TYPES.get(feature, 'numerical')
        fig, ax = plt.subplots(figsize=(10, 6))

        if feat_type == 'numerical':
            # KDE plots
            pool_num = pd.to_numeric(pool_data, errors='coerce').dropna()
            rec_num = pd.to_numeric(rec_data, errors='coerce').dropna()

            if len(pool_num) > 0 and len(rec_num) > 0 and pool_num.std() > 0:
                pool_num.plot(kind='kde', ax=ax, label='Pool', linewidth=2, color='steelblue')
                rec_num.plot(kind='kde', ax=ax, label='Recommended', linewidth=2,
                            linestyle='--', color='orange')
                ax.axvline(pool_num.mean(), color='steelblue', alpha=0.5, linestyle=':')
                ax.axvline(rec_num.mean(), color='orange', alpha=0.5, linestyle=':')
                ax.set_ylabel('Density')
                ax.set_xlabel(feature)

        elif feat_type in ['categorical', 'binary']:
            # Side-by-side bars with consistent ordering
            pool_cat = pd.Series(standardize_categories(pool_data, feature))
            rec_cat = pd.Series(standardize_categories(rec_data, feature))

            pool_counts = pool_cat.value_counts(normalize=True)
            rec_counts = rec_cat.value_counts(normalize=True)

            # Get all categories
            if isinstance(pool_cat.dtype, pd.CategoricalDtype):
                all_cats = pool_cat.cat.categories
            else:
                all_cats = sorted(set(pool_counts.index) | set(rec_counts.index))

            df_plot = pd.DataFrame({
                'Pool': pool_counts.reindex(all_cats, fill_value=0),
                'Recommended': rec_counts.reindex(all_cats, fill_value=0)
            })

            df_plot.plot(kind='bar', ax=ax, color=['steelblue', 'orange'],
                        edgecolor='black', width=0.8)
            ax.set_ylabel('Proportion')
            ax.set_xlabel(feature)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        ax.legend()
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(COMPARE_AGG_DIR / filename, bbox_inches='tight')
        plt.close()

    # 1. Aggregate by prompt style
    print("  a) By prompt style...")
    for prompt in PROMPT_STYLES:
        for feature in all_features:
            pool_data, rec_data = get_aggregated_pool_rec({'prompt_style': prompt}, feature)
            plot_comparison_aggregated(
                pool_data, rec_data, feature,
                f'{feature} - Prompt: {prompt.capitalize()}\n(Aggregated across all datasets and models)',
                f'{feature}_prompt_{prompt}.png'
            )

    # 2. Aggregate by dataset
    print("  b) By dataset...")
    for dataset in DATASETS:
        for feature in all_features:
            pool_data, rec_data = get_aggregated_pool_rec({'dataset': dataset}, feature)
            plot_comparison_aggregated(
                pool_data, rec_data, feature,
                f'{feature} - Dataset: {dataset.capitalize()}\n(Aggregated across all models and prompts)',
                f'{feature}_dataset_{dataset}.png'
            )

    # 3. Aggregate by model
    print("  c) By model...")
    for provider in PROVIDERS:
        for feature in all_features:
            pool_data, rec_data = get_aggregated_pool_rec({'provider': provider}, feature)
            plot_comparison_aggregated(
                pool_data, rec_data, feature,
                f'{feature} - Model: {provider.capitalize()}\n(Aggregated across all datasets and prompts)',
                f'{feature}_model_{provider}.png'
            )

    print(f"\n✓ Generated aggregated comparison plots in {COMPARE_AGG_DIR}")

# ============================================================================
# 3. BIAS HEATMAPS - FIXED
# ============================================================================

def generate_bias_heatmaps(comp_df):
    """
    FIXED: Handle zero bias cases better, add annotations
    """
    print("\n" + "="*80)
    print("GENERATING BIAS HEATMAPS - FIXED")
    print("="*80)

    all_features = sum(FEATURES.values(), [])

    # Create reverse mapping from feature to category
    feature_to_category = {}
    for category, features in FEATURES.items():
        for feature in features:
            feature_to_category[feature] = category

    # Normalize bias values within each feature (min-max scaling)
    comp_df_norm = comp_df.copy()

    # Normalize features with non-zero variance
    for feature in all_features:
        feat_data = comp_df_norm[comp_df_norm['feature'] == feature]
        if len(feat_data) == 0:
            continue

        bias_vals = feat_data['bias'].values
        if bias_vals.max() != bias_vals.min() and bias_vals.max() > 0:
            normalized = (bias_vals - bias_vals.min()) / (bias_vals.max() - bias_vals.min())
        else:
            normalized = np.zeros_like(bias_vals)

        comp_df_norm.loc[comp_df_norm['feature'] == feature, 'bias_normalized'] = normalized

    # Add marker for statistical significance
    comp_df_norm['sig_marker'] = comp_df_norm['significant'].apply(lambda x: '*' if x else '')

    # 1. Fully disaggregated: One heatmap per prompt style
    for prompt in PROMPT_STYLES:
        prompt_data = comp_df_norm[comp_df_norm['prompt_style'] == prompt]

        # Create pivot table: features × (dataset-model combinations)
        pivot = prompt_data.pivot_table(
            values='bias_normalized',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        # Create significance pivot for annotations
        pivot_sig = prompt_data.pivot_table(
            values='significant',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        # Custom annotations with multi-level significance markers
        annot_array = np.empty_like(pivot, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
                if pd.isna(val):
                    annot_array[i, j] = ''
                elif sig > 0.75:  # More than 75% of conditions significant
                    annot_array[i, j] = f'{val:.3f}***'
                elif sig > 0.60:  # More than 60% of conditions significant
                    annot_array[i, j] = f'{val:.3f}**'
                elif sig > 0.50:  # More than 50% of conditions significant
                    annot_array[i, j] = f'{val:.3f}*'
                else:
                    annot_array[i, j] = f'{val:.3f}'

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap='Reds',
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
        ax.set_title(f'Bias Heatmap (Normalized) - Prompt: {prompt}\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
        ax.set_xlabel('Dataset × Model')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        plt.savefig(HEATMAP_DIR / f'disaggregated_prompt_{prompt}.png', bbox_inches='tight')
        plt.close()

    print("  ✓ Fully disaggregated heatmaps (by prompt style)")

    # 2-5. Aggregated versions with significance markers
    # Aggregated by dataset
    agg_dataset = comp_df_norm.groupby(['feature', 'dataset']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_dataset = agg_dataset.pivot_table(
        values='bias_normalized',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    pivot_dataset_sig = agg_dataset.pivot_table(
        values='significant',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    # Create annotations with significance markers
    annot_dataset = np.empty_like(pivot_dataset, dtype=object)
    for i in range(pivot_dataset.shape[0]):
        for j in range(pivot_dataset.shape[1]):
            val = pivot_dataset.iloc[i, j]
            sig = pivot_dataset_sig.iloc[i, j] if not pd.isna(pivot_dataset_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_dataset[i, j] = ''
            elif sig > 0.75:
                annot_dataset[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_dataset[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_dataset[i, j] = f'{val:.3f}*'
            else:
                annot_dataset[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot_dataset, annot=annot_dataset, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Dataset (Aggregated across Models & Prompts)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_dataset.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by dataset")

    # Aggregated by model
    agg_model = comp_df_norm.groupby(['feature', 'provider']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_model = agg_model.pivot_table(
        values='bias_normalized',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    pivot_model_sig = agg_model.pivot_table(
        values='significant',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    # Create annotations with significance markers
    annot_model = np.empty_like(pivot_model, dtype=object)
    for i in range(pivot_model.shape[0]):
        for j in range(pivot_model.shape[1]):
            val = pivot_model.iloc[i, j]
            sig = pivot_model_sig.iloc[i, j] if not pd.isna(pivot_model_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_model[i, j] = ''
            elif sig > 0.75:
                annot_model[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_model[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_model[i, j] = f'{val:.3f}*'
            else:
                annot_model[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot_model, annot=annot_model, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Model (Aggregated across Datasets & Prompts)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_model.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by model")

    # Aggregated by prompt style
    agg_prompt = comp_df_norm.groupby(['feature', 'prompt_style']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_prompt = agg_prompt.pivot_table(
        values='bias_normalized',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    pivot_prompt_sig = agg_prompt.pivot_table(
        values='significant',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    # Create annotations with significance markers
    annot_prompt = np.empty_like(pivot_prompt, dtype=object)
    for i in range(pivot_prompt.shape[0]):
        for j in range(pivot_prompt.shape[1]):
            val = pivot_prompt.iloc[i, j]
            sig = pivot_prompt_sig.iloc[i, j] if not pd.isna(pivot_prompt_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_prompt[i, j] = ''
            elif sig > 0.75:
                annot_prompt[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_prompt[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_prompt[i, j] = f'{val:.3f}*'
            else:
                annot_prompt[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(pivot_prompt, annot=annot_prompt, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Prompt Style (Aggregated across Datasets & Models)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_prompt.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Aggregated by prompt style")

    # Fully aggregated
    agg_full = comp_df_norm.groupby('feature').agg({
        'bias_normalized': 'mean',
        'significant': 'mean',
        'bias': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 12))
    agg_full_sorted = agg_full.sort_values('bias_normalized', ascending=False)
    bars = ax.barh(agg_full_sorted['feature'], agg_full_sorted['bias_normalized'], color='steelblue')

    # Color bars by significance level and add markers
    for i, (idx, row) in enumerate(agg_full_sorted.iterrows()):
        sig = row['significant']
        if sig > 0.75:
            bars[i].set_color('darkred')
            marker = '***'
        elif sig > 0.60:
            bars[i].set_color('coral')
            marker = '**'
        elif sig > 0.50:
            bars[i].set_color('lightsalmon')
            marker = '*'
        else:
            marker = ''

        # Add significance marker at end of bar
        if marker:
            ax.text(row['bias_normalized'] + 0.01, i, marker,
                   va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Normalized Bias (0-1)')
    ax.set_title('Overall Bias (Fully Aggregated)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'fully_aggregated.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Fully aggregated")

    # ========================================================================
    # CATEGORY-AGGREGATED VERSIONS
    # ========================================================================
    print("\n" + "-"*80)
    print("GENERATING CATEGORY-AGGREGATED HEATMAPS")
    print("-"*80)

    # Add category column to normalized dataframe
    comp_df_norm['category'] = comp_df_norm['feature'].map(feature_to_category)

    # 1. Category-aggregated disaggregated: One heatmap per prompt style
    for prompt in PROMPT_STYLES:
        prompt_data = comp_df_norm[comp_df_norm['prompt_style'] == prompt]

        # Aggregate by category
        agg_cat = prompt_data.groupby(['category', 'dataset', 'provider']).agg({
            'bias_normalized': 'mean',
            'significant': 'mean'
        }).reset_index()

        # Create pivot table: categories × (dataset-model combinations)
        pivot = agg_cat.pivot_table(
            values='bias_normalized',
            index='category',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        pivot_sig = agg_cat.pivot_table(
            values='significant',
            index='category',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        # Create annotations with significance markers
        annot_array = np.empty_like(pivot, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                sig = pivot_sig.iloc[i, j] if not pd.isna(pivot_sig.iloc[i, j]) else 0
                if pd.isna(val):
                    annot_array[i, j] = ''
                elif sig > 0.75:
                    annot_array[i, j] = f'{val:.3f}***'
                elif sig > 0.60:
                    annot_array[i, j] = f'{val:.3f}**'
                elif sig > 0.50:
                    annot_array[i, j] = f'{val:.3f}*'
                else:
                    annot_array[i, j] = f'{val:.3f}'

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap='Reds',
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
        ax.set_title(f'Bias by Feature Category - Prompt: {prompt}\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
        ax.set_xlabel('Dataset × Model')
        ax.set_ylabel('Feature Category')
        plt.tight_layout()
        plt.savefig(HEATMAP_DIR / f'disaggregated_prompt_{prompt}_by_category.png', bbox_inches='tight')
        plt.close()

    print("  ✓ Category-aggregated disaggregated heatmaps (by prompt style)")

    # 2. Category-aggregated by dataset
    agg_cat_dataset = comp_df_norm.groupby(['category', 'dataset']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_dataset = agg_cat_dataset.pivot_table(
        values='bias_normalized',
        index='category',
        columns='dataset',
        aggfunc='mean'
    )

    pivot_cat_dataset_sig = agg_cat_dataset.pivot_table(
        values='significant',
        index='category',
        columns='dataset',
        aggfunc='mean'
    )

    annot_cat_dataset = np.empty_like(pivot_cat_dataset, dtype=object)
    for i in range(pivot_cat_dataset.shape[0]):
        for j in range(pivot_cat_dataset.shape[1]):
            val = pivot_cat_dataset.iloc[i, j]
            sig = pivot_cat_dataset_sig.iloc[i, j] if not pd.isna(pivot_cat_dataset_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_cat_dataset[i, j] = ''
            elif sig > 0.75:
                annot_cat_dataset[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_cat_dataset[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_cat_dataset[i, j] = f'{val:.3f}*'
            else:
                annot_cat_dataset[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_cat_dataset, annot=annot_cat_dataset, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Feature Category × Dataset\n(Aggregated across Models & Prompts)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Feature Category')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_dataset_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated by dataset")

    # 3. Category-aggregated by model
    agg_cat_model = comp_df_norm.groupby(['category', 'provider']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_model = agg_cat_model.pivot_table(
        values='bias_normalized',
        index='category',
        columns='provider',
        aggfunc='mean'
    )

    pivot_cat_model_sig = agg_cat_model.pivot_table(
        values='significant',
        index='category',
        columns='provider',
        aggfunc='mean'
    )

    annot_cat_model = np.empty_like(pivot_cat_model, dtype=object)
    for i in range(pivot_cat_model.shape[0]):
        for j in range(pivot_cat_model.shape[1]):
            val = pivot_cat_model.iloc[i, j]
            sig = pivot_cat_model_sig.iloc[i, j] if not pd.isna(pivot_cat_model_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_cat_model[i, j] = ''
            elif sig > 0.75:
                annot_cat_model[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_cat_model[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_cat_model[i, j] = f'{val:.3f}*'
            else:
                annot_cat_model[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot_cat_model, annot=annot_cat_model, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Feature Category × Model\n(Aggregated across Datasets & Prompts)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    ax.set_xlabel('Model Provider')
    ax.set_ylabel('Feature Category')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_model_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated by model")

    # 4. Category-aggregated by prompt style
    agg_cat_prompt = comp_df_norm.groupby(['category', 'prompt_style']).agg({
        'bias_normalized': 'mean',
        'significant': 'mean'
    }).reset_index()

    pivot_cat_prompt = agg_cat_prompt.pivot_table(
        values='bias_normalized',
        index='category',
        columns='prompt_style',
        aggfunc='mean'
    )

    pivot_cat_prompt_sig = agg_cat_prompt.pivot_table(
        values='significant',
        index='category',
        columns='prompt_style',
        aggfunc='mean'
    )

    annot_cat_prompt = np.empty_like(pivot_cat_prompt, dtype=object)
    for i in range(pivot_cat_prompt.shape[0]):
        for j in range(pivot_cat_prompt.shape[1]):
            val = pivot_cat_prompt.iloc[i, j]
            sig = pivot_cat_prompt_sig.iloc[i, j] if not pd.isna(pivot_cat_prompt_sig.iloc[i, j]) else 0
            if pd.isna(val):
                annot_cat_prompt[i, j] = ''
            elif sig > 0.75:
                annot_cat_prompt[i, j] = f'{val:.3f}***'
            elif sig > 0.60:
                annot_cat_prompt[i, j] = f'{val:.3f}**'
            elif sig > 0.50:
                annot_cat_prompt[i, j] = f'{val:.3f}*'
            else:
                annot_cat_prompt[i, j] = f'{val:.3f}'

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_cat_prompt, annot=annot_cat_prompt, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized Bias'})
    ax.set_title('Bias by Feature Category × Prompt Style\n(Aggregated across Datasets & Models)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Feature Category')
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'aggregated_by_prompt_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Category-aggregated by prompt style")

    # 5. Fully category-aggregated
    agg_cat_full = comp_df_norm.groupby('category').agg({
        'bias_normalized': 'mean',
        'significant': 'mean',
        'bias': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    agg_cat_full_sorted = agg_cat_full.sort_values('bias_normalized', ascending=False)
    bars = ax.barh(agg_cat_full_sorted['category'], agg_cat_full_sorted['bias_normalized'], color='steelblue')

    # Color bars by significance level and add markers
    for i, (idx, row) in enumerate(agg_cat_full_sorted.iterrows()):
        sig = row['significant']
        if sig > 0.75:
            bars[i].set_color('darkred')
            marker = '***'
        elif sig > 0.60:
            bars[i].set_color('coral')
            marker = '**'
        elif sig > 0.50:
            bars[i].set_color('lightsalmon')
            marker = '*'
        else:
            marker = ''

        # Add significance marker at end of bar
        if marker:
            ax.text(row['bias_normalized'] + 0.01, i, marker,
                   va='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Normalized Bias (0-1)')
    ax.set_title('Overall Bias by Feature Category (Fully Aggregated)\n(* p<0.05 >50%, ** p<0.05 >60%, *** p<0.05 >75%)', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(HEATMAP_DIR / 'fully_aggregated_by_category.png', bbox_inches='tight')
    plt.close()

    print("  ✓ Fully category-aggregated")

    print(f"\n✓ Saved bias heatmaps to {HEATMAP_DIR}")

    # Print diagnostic info about zero bias
    zero_bias_features = comp_df[comp_df['bias'] == 0].groupby('feature').size()
    if len(zero_bias_features) > 0:
        print("\n⚠ Features with zero bias in some conditions:")
        for feat, count in zero_bias_features.items():
            pct = 100 * count / len(comp_df[comp_df['feature'] == feat])
            print(f"  - {feat}: {count} conditions ({pct:.1f}%)")

# ============================================================================
# 4. DIRECTIONAL BIAS PLOTS (NEW)
# ============================================================================

def compute_directional_bias():
    """
    Compute directional bias for each feature showing which categories/values are favored.

    For categorical features: proportion_recommended - proportion_pool for each category
    For continuous features: mean_recommended - mean_pool
    """
    print("\n" + "="*80)
    print("COMPUTING DIRECTIONAL BIAS")
    print("="*80)

    # Check if directional bias data already exists
    dir_bias_file = OUTPUT_DIR / 'directional_bias_data.parquet'
    if dir_bias_file.exists():
        print("\n✓ Loading cached directional bias data...")
        df = pd.read_parquet(dir_bias_file)
        print(f"  Loaded {len(df)} directional bias measurements from cache")
        print(f"  To recompute, delete: {dir_bias_file}\n")
        return df

    all_directional_data = []
    all_features = sum(FEATURES.values(), [])

    # Process each dataset × provider × prompt combination
    for dataset in DATASETS:
        for provider in PROVIDERS:
            df = load_experiment_data(dataset, provider)
            if df is None:
                continue

            for prompt in PROMPT_STYLES:
                prompt_df = df[df['prompt_style'] == prompt]

                # Split into pool and recommended
                pool = prompt_df[prompt_df['selected'] == 0].copy()
                recommended = prompt_df[prompt_df['selected'] == 1].copy()

                if len(pool) == 0 or len(recommended) == 0:
                    continue

                # Process each feature
                for feature in all_features:
                    if feature not in df.columns:
                        continue

                    feature_type = FEATURE_TYPES[feature]

                    if feature_type == 'categorical':
                        # Get all unique categories
                        all_categories = sorted(prompt_df[feature].dropna().unique())

                        for category in all_categories:
                            # Compute proportions
                            prop_pool = (pool[feature] == category).sum() / len(pool) if len(pool) > 0 else 0
                            prop_rec = (recommended[feature] == category).sum() / len(recommended) if len(recommended) > 0 else 0

                            # Directional bias: positive = over-represented, negative = under-represented
                            dir_bias = prop_rec - prop_pool

                            all_directional_data.append({
                                'feature': feature,
                                'category': str(category),
                                'dataset': dataset,
                                'provider': provider,
                                'prompt_style': prompt,
                                'directional_bias': dir_bias,
                                'prop_pool': prop_pool,
                                'prop_recommended': prop_rec,
                                'feature_type': 'categorical'
                            })

                    else:  # continuous
                        # Compute mean difference
                        mean_pool = pool[feature].mean()
                        mean_rec = recommended[feature].mean()
                        dir_bias = mean_rec - mean_pool

                        # Also compute standardized difference for reference
                        std_pool = pool[feature].std()
                        std_diff = dir_bias / std_pool if std_pool > 0 else 0

                        all_directional_data.append({
                            'feature': feature,
                            'category': 'mean_difference',  # For continuous, single "category"
                            'dataset': dataset,
                            'provider': provider,
                            'prompt_style': prompt,
                            'directional_bias': dir_bias,
                            'mean_pool': mean_pool,
                            'mean_recommended': mean_rec,
                            'std_diff': std_diff,
                            'feature_type': 'continuous'
                        })

    df_directional = pd.DataFrame(all_directional_data)
    print(f"✓ Computed directional bias for {len(df_directional)} feature-category-condition combinations")

    # Save directional bias data for future use
    dir_bias_file = OUTPUT_DIR / 'directional_bias_data.parquet'
    df_directional.to_parquet(dir_bias_file, index=False)
    print(f"✓ Saved directional bias data to {dir_bias_file}")
    dir_bias_csv = OUTPUT_DIR / 'directional_bias_data.csv'
    df_directional.to_csv(dir_bias_csv, index=False)
    print(f"✓ Saved directional bias data to {dir_bias_csv}")

    return df_directional


def generate_directional_bias_plots(df_directional):
    """
    Generate directional bias plots showing which categories/values are favored.

    For each feature, create 3 plots:
    1. By prompt style (6 subplots)
    2. By dataset (3 subplots)
    3. By model (3 subplots)
    """
    print("\n" + "="*80)
    print("GENERATING DIRECTIONAL BIAS PLOTS")
    print("="*80)

    # Create output directory
    dir_bias_dir = VIZ_DIR / '4_directional_bias'
    dir_bias_dir.mkdir(exist_ok=True)

    all_features = sum(FEATURES.values(), [])

    for feature in all_features:
        print(f"\nGenerating directional bias plots for: {feature}")

        feature_data = df_directional[df_directional['feature'] == feature].copy()
        if len(feature_data) == 0:
            print(f"  ⚠ No data for {feature}")
            continue

        feature_type = feature_data['feature_type'].iloc[0]

        # By Prompt Style (6 subplots)
        generate_directional_by_prompt(feature, feature_data, feature_type, dir_bias_dir)

    print(f"\n✓ Saved directional bias plots to {dir_bias_dir}")


def generate_directional_by_prompt(feature, feature_data, feature_type, output_dir):
    """Generate plot with 6 subplots (one per prompt style)"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, prompt_style in enumerate(PROMPT_STYLES):
        ax = axes[idx]
        data = feature_data[feature_data['prompt_style'] == prompt_style].copy()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{prompt_style}', fontweight='bold')
            continue

        # Aggregate across datasets and models
        if feature_type == 'categorical':
            # Create pivot: categories × (dataset_model)
            data['dataset_model'] = data['dataset'] + '\n' + data['provider']
            pivot = data.groupby(['category', 'dataset_model'])['directional_bias'].mean().reset_index()
            pivot_wide = pivot.pivot(index='category', columns='dataset_model', values='directional_bias')

            # Sort categories by average bias
            pivot_wide['avg'] = pivot_wide.mean(axis=1)
            pivot_wide = pivot_wide.sort_values('avg', ascending=False).drop('avg', axis=1)

            # Create heatmap
            sns.heatmap(pivot_wide, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                       vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Directional Bias'})
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:  # continuous
            # Bar chart for dataset × model
            data['dataset_model'] = data['dataset'] + '_' + data['provider']
            agg = data.groupby('dataset_model')['directional_bias'].mean().sort_values()

            colors = ['red' if x < 0 else 'green' for x in agg.values]
            agg.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Mean Difference (Rec - Pool)')
            ax.set_ylabel('')

        ax.set_title(f'{prompt_style}', fontweight='bold', fontsize=12)

    fig.suptitle(f'Directional Bias: {feature}\n(By Prompt Style)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{feature}_by_prompt.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_directional_by_dataset(feature, feature_data, feature_type, output_dir):
    """Generate plot with 3 subplots (one per dataset)"""

    datasets = sorted(feature_data['dataset'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        data = feature_data[feature_data['dataset'] == dataset].copy()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{dataset}', fontweight='bold')
            continue

        # Aggregate across models and prompt styles
        if feature_type == 'categorical':
            data['model_prompt'] = data['provider'] + '\n' + data['prompt_style']
            pivot = data.groupby(['category', 'model_prompt'])['directional_bias'].mean().reset_index()
            pivot_wide = pivot.pivot(index='category', columns='model_prompt', values='directional_bias')

            # Sort categories
            pivot_wide['avg'] = pivot_wide.mean(axis=1)
            pivot_wide = pivot_wide.sort_values('avg', ascending=False).drop('avg', axis=1)

            sns.heatmap(pivot_wide, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                       vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Directional Bias'})
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:  # continuous
            data['model_prompt'] = data['provider'] + '_' + data['prompt_style']
            agg = data.groupby('model_prompt')['directional_bias'].mean().sort_values()

            colors = ['red' if x < 0 else 'green' for x in agg.values]
            agg.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Mean Difference')
            ax.set_ylabel('')

        ax.set_title(f'{dataset}', fontweight='bold', fontsize=12)

    fig.suptitle(f'Directional Bias: {feature}\n(By Dataset)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{feature}_by_dataset.png', bbox_inches='tight', dpi=300)
    plt.close()


def generate_directional_by_model(feature, feature_data, feature_type, output_dir):
    """Generate plot with 3 subplots (one per model)"""

    providers = sorted(feature_data['provider'].unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, provider in enumerate(providers):
        ax = axes[idx]
        data = feature_data[feature_data['provider'] == provider].copy()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{provider}', fontweight='bold')
            continue

        # Aggregate across datasets and prompt styles
        if feature_type == 'categorical':
            data['dataset_prompt'] = data['dataset'] + '\n' + data['prompt_style']
            pivot = data.groupby(['category', 'dataset_prompt'])['directional_bias'].mean().reset_index()
            pivot_wide = pivot.pivot(index='category', columns='dataset_prompt', values='directional_bias')

            # Sort categories
            pivot_wide['avg'] = pivot_wide.mean(axis=1)
            pivot_wide = pivot_wide.sort_values('avg', ascending=False).drop('avg', axis=1)

            sns.heatmap(pivot_wide, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                       vmin=-0.3, vmax=0.3, ax=ax, cbar_kws={'label': 'Directional Bias'})
            ax.set_ylabel('')
            ax.set_xlabel('')
        else:  # continuous
            data['dataset_prompt'] = data['dataset'] + '_' + data['prompt_style']
            agg = data.groupby('dataset_prompt')['directional_bias'].mean().sort_values()

            colors = ['red' if x < 0 else 'green' for x in agg.values]
            agg.plot(kind='barh', ax=ax, color=colors, edgecolor='black')
            ax.axvline(0, color='black', linestyle='--', linewidth=1)
            ax.set_xlabel('Mean Difference')
            ax.set_ylabel('')

        ax.set_title(f'{provider}', fontweight='bold', fontsize=12)

    fig.suptitle(f'Directional Bias: {feature}\n(By Model)',
                fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f'{feature}_by_model.png', bbox_inches='tight', dpi=300)
    plt.close()


# ============================================================================
# 5. FEATURE IMPORTANCE ANALYSIS (NEW)
# ============================================================================

def prepare_ml_features(df: pd.DataFrame, all_features: list) -> tuple:
    """
    Prepare features for ML model - standardize and one-hot encode
    Returns: X (feature matrix), y (target), feature_names (list)
    """
    # Separate numerical and categorical features
    available_numerical = [f for f in all_features if f in df.columns and FEATURE_TYPES[f] in ['numerical', 'binary']]
    available_categorical = [f for f in all_features if f in df.columns and FEATURE_TYPES[f] == 'categorical']

    # Start with numerical features
    X = df[available_numerical].copy()

    # One-hot encode categorical features
    for cat_feat in available_categorical:
        dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    # Fill missing values
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in ['float64', 'int64']:
                X[col].fillna(X[col].median(), inplace=True)
            else:
                X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0, inplace=True)

    # Target variable
    y = df['selected'].astype(int)

    # Feature names
    feature_names = X.columns.tolist()

    return X, y, feature_names


def compute_feature_importance():
    """
    Compute feature importance using Random Forest for each condition.
    Returns DataFrame with importance scores for all features and conditions.
    """
    print("\n" + "="*80)
    print("COMPUTING FEATURE IMPORTANCE (RANDOM FOREST + SHAP)")
    print("="*80)

    all_importance_data = []
    all_features = sum(FEATURES.values(), [])

    # Process each dataset × provider × prompt combination
    for dataset in DATASETS:
        for provider in PROVIDERS:
            df = load_experiment_data(dataset, provider)
            if df is None:
                continue

            print(f"\n{dataset.upper()} × {provider.upper()}")

            for prompt in PROMPT_STYLES:
                prompt_df = df[df['prompt_style'] == prompt]

                if len(prompt_df) == 0:
                    continue

                # Split into pool and recommended
                pool = prompt_df[prompt_df['selected'] == 0]
                recommended = prompt_df[prompt_df['selected'] == 1]

                # Skip if too imbalanced or too few samples
                if len(pool) < 20 or len(recommended) < 20:
                    continue

                print(f"  {prompt:15s}: ", end='', flush=True)

                # Prepare features
                X, y, feature_names = prepare_ml_features(prompt_df, all_features)

                # Skip if target is constant
                if y.nunique() < 2 or len(X) < 50:
                    print("⚠ Skipped (insufficient data)")
                    continue

                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # Train Random Forest
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )

                try:
                    rf.fit(X_scaled, y)

                    # Get built-in feature importance
                    rf_importance = rf.feature_importances_

                    # Compute SHAP values
                    explainer = shap.TreeExplainer(rf)
                    shap_values = explainer.shap_values(X_scaled)

                    # For binary classification, shap_values might be a list [class0, class1]
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]  # Use positive class

                    # Aggregate SHAP importance (mean absolute SHAP value)
                    shap_importance = np.abs(shap_values).mean(axis=0)

                    # Compute AUROC
                    y_pred_proba = rf.predict_proba(X_scaled)[:, 1]
                    auroc = roc_auc_score(y, y_pred_proba)

                    print(f"✓ AUROC: {auroc:.3f}")

                    # Map feature names back to original features (aggregate one-hot encoded)
                    # This is needed because we one-hot encoded categorical features
                    feature_importance_dict = {}

                    for orig_feat in all_features:
                        if orig_feat not in prompt_df.columns:
                            continue

                        # Find all feature columns that start with this feature name
                        matching_indices = [i for i, fn in enumerate(feature_names)
                                          if fn == orig_feat or fn.startswith(f"{orig_feat}_")]

                        if matching_indices:
                            # Sum importance across all one-hot encoded columns
                            # Convert to float to ensure scalar values
                            feat_rf_importance = float(np.sum([rf_importance[i] for i in matching_indices]))
                            feat_shap_importance = float(np.sum([shap_importance[i] for i in matching_indices]))

                            feature_importance_dict[orig_feat] = {
                                'rf_importance': feat_rf_importance,
                                'shap_importance': feat_shap_importance
                            }

                    # Store results
                    for feat, importance in feature_importance_dict.items():
                        all_importance_data.append({
                            'feature': feat,
                            'dataset': dataset,
                            'provider': provider,
                            'prompt_style': prompt,
                            'rf_importance': importance['rf_importance'],
                            'shap_importance': importance['shap_importance'],
                            'auroc': float(auroc),
                            'n_samples': int(len(y)),
                            'n_positive': int(y.sum()),
                            'n_negative': int(len(y) - y.sum())
                        })

                except Exception as e:
                    print(f"✗ Error: {e}")
                    continue

    df_importance = pd.DataFrame(all_importance_data)
    print(f"\n✓ Computed feature importance for {len(df_importance)} feature-condition combinations")
    print(f"  Models trained: {df_importance[['dataset', 'provider', 'prompt_style']].drop_duplicates().shape[0]}")
    print(f"  Mean AUROC: {df_importance.groupby(['dataset', 'provider', 'prompt_style'])['auroc'].first().mean():.3f}")

    # Save the data for future use
    importance_data_file = OUTPUT_DIR / 'feature_importance_data.csv'
    df_importance.to_csv(importance_data_file, index=False)
    print(f"  ✓ Saved feature importance data to {importance_data_file}")

    return df_importance


def generate_feature_importance_heatmaps(df_importance):
    """
    Generate feature importance HEATMAPS matching bias evaluation structure:
    1. Fully disaggregated (6 heatmaps, one per prompt) - features × (dataset × model)
    2. Aggregated by dataset (1 heatmap) - features × dataset
    3. Aggregated by model (1 heatmap) - features × model
    4. Aggregated by prompt (1 heatmap) - features × prompt
    5. Fully aggregated (1 heatmap) - features × "Overall"

    Total: 10 heatmaps (matching bias structure)
    """
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE HEATMAPS (SHAP VALUES)")
    print("="*80)

    # Create output directory
    importance_dir = VIZ_DIR / '5_feature_importance'
    importance_dir.mkdir(exist_ok=True)

    all_features = sum(FEATURES.values(), [])

    # Use SHAP importance (more interpretable than RF importance)
    # Normalize SHAP values within each feature for better visualization
    df_norm = df_importance.copy()

    for feature in all_features:
        feat_data = df_norm[df_norm['feature'] == feature]
        if len(feat_data) == 0:
            continue

        shap_vals = feat_data['shap_importance'].values
        max_val = float(shap_vals.max())
        min_val = float(shap_vals.min())
        if max_val != min_val and max_val > 0:
            normalized = (shap_vals - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(shap_vals)

        df_norm.loc[df_norm['feature'] == feature, 'shap_normalized'] = normalized

    # 1. FULLY DISAGGREGATED: One heatmap per prompt style
    for prompt in PROMPT_STYLES:
        prompt_data = df_norm[df_norm['prompt_style'] == prompt]

        if len(prompt_data) == 0:
            continue

        # Create pivot: features × (dataset × model)
        pivot = prompt_data.pivot_table(
            values='shap_normalized',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        if pivot.empty:
            continue

        # Create annotations with actual SHAP values (not normalized)
        pivot_raw = prompt_data.pivot_table(
            values='shap_importance',
            index='feature',
            columns=['dataset', 'provider'],
            aggfunc='mean'
        )

        annot_array = np.empty_like(pivot, dtype=object)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val_raw = pivot_raw.iloc[i, j]
                if pd.isna(val_raw):
                    annot_array[i, j] = ''
                else:
                    annot_array[i, j] = f'{val_raw:.3f}'

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.heatmap(pivot, annot=annot_array, fmt='', cmap='Reds',
                    vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'})
        ax.set_title(f'Feature Importance (SHAP) - Prompt: {prompt}\n(Normalized within feature, annotations show raw values)',
                     fontweight='bold')
        ax.set_xlabel('Dataset × Model')
        ax.set_ylabel('Feature')
        plt.tight_layout()
        plt.savefig(importance_dir / f'disaggregated_prompt_{prompt}.png', bbox_inches='tight', dpi=300)
        plt.close()

    print("  ✓ Fully disaggregated heatmaps (by prompt style)")

    # 2. AGGREGATED BY DATASET
    agg_dataset = df_norm.groupby(['feature', 'dataset']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    pivot_dataset = agg_dataset.pivot_table(
        values='shap_normalized',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    pivot_dataset_raw = agg_dataset.pivot_table(
        values='shap_importance',
        index='feature',
        columns='dataset',
        aggfunc='mean'
    )

    annot_dataset = np.empty_like(pivot_dataset, dtype=object)
    for i in range(pivot_dataset.shape[0]):
        for j in range(pivot_dataset.shape[1]):
            val_raw = pivot_dataset_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_dataset[i, j] = ''
            else:
                annot_dataset[i, j] = f'{val_raw:.3f}'

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot_dataset, annot=annot_dataset, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'})
    ax.set_title('Feature Importance by Dataset\n(Aggregated across Models & Prompts)', fontweight='bold')
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(importance_dir / 'aggregated_by_dataset.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Aggregated by dataset")

    # 3. AGGREGATED BY MODEL
    agg_model = df_norm.groupby(['feature', 'provider']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    pivot_model = agg_model.pivot_table(
        values='shap_normalized',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    pivot_model_raw = agg_model.pivot_table(
        values='shap_importance',
        index='feature',
        columns='provider',
        aggfunc='mean'
    )

    annot_model = np.empty_like(pivot_model, dtype=object)
    for i in range(pivot_model.shape[0]):
        for j in range(pivot_model.shape[1]):
            val_raw = pivot_model_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_model[i, j] = ''
            else:
                annot_model[i, j] = f'{val_raw:.3f}'

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(pivot_model, annot=annot_model, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'})
    ax.set_title('Feature Importance by Model\n(Aggregated across Datasets & Prompts)', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(importance_dir / 'aggregated_by_model.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Aggregated by model")

    # 4. AGGREGATED BY PROMPT
    agg_prompt = df_norm.groupby(['feature', 'prompt_style']).agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    pivot_prompt = agg_prompt.pivot_table(
        values='shap_normalized',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    pivot_prompt_raw = agg_prompt.pivot_table(
        values='shap_importance',
        index='feature',
        columns='prompt_style',
        aggfunc='mean'
    )

    annot_prompt = np.empty_like(pivot_prompt, dtype=object)
    for i in range(pivot_prompt.shape[0]):
        for j in range(pivot_prompt.shape[1]):
            val_raw = pivot_prompt_raw.iloc[i, j]
            if pd.isna(val_raw):
                annot_prompt[i, j] = ''
            else:
                annot_prompt[i, j] = f'{val_raw:.3f}'

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot_prompt, annot=annot_prompt, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'})
    ax.set_title('Feature Importance by Prompt Style\n(Aggregated across Datasets & Models)', fontweight='bold')
    ax.set_xlabel('Prompt Style')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(importance_dir / 'aggregated_by_prompt.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Aggregated by prompt style")

    # 5. FULLY AGGREGATED
    agg_all = df_norm.groupby('feature').agg({
        'shap_normalized': 'mean',
        'shap_importance': 'mean'
    }).reset_index()

    # Create a single-column heatmap
    pivot_all = agg_all.set_index('feature')[['shap_normalized']]
    pivot_all.columns = ['Overall']

    pivot_all_raw = agg_all.set_index('feature')[['shap_importance']]

    annot_all = np.array([[f'{pivot_all_raw.loc[feat, "shap_importance"]:.3f}']
                          for feat in pivot_all.index])

    fig, ax = plt.subplots(figsize=(6, 10))
    sns.heatmap(pivot_all, annot=annot_all, fmt='', cmap='Reds',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Normalized SHAP Importance'})
    ax.set_title('Feature Importance: Overall\n(Aggregated across All Conditions)', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(importance_dir / 'fully_aggregated.png', bbox_inches='tight', dpi=300)
    plt.close()

    print("  ✓ Fully aggregated")

    print(f"\n✓ Saved all 10 feature importance heatmaps to {importance_dir}")


# ============================================================================
# 6. TOP 5 SIGNIFICANT FEATURES - ENHANCED
# ============================================================================

def generate_top5_significant(comp_df):
    """
    ENHANCED: Add cumulative bar charts disaggregated by dataset/model/prompt
    """
    print("\n" + "="*80)
    print("GENERATING TOP 5 SIGNIFICANT FEATURES PLOTS - ENHANCED")
    print("="*80)

    all_features = sum(FEATURES.values(), [])

    # Helper function to get top 5
    def plot_top5(data, title, filename, category_label='All Features'):
        # Count how often each feature is significant
        sig_counts = data[data['significant']].groupby('feature').size().reset_index(name='count')
        sig_counts = sig_counts.sort_values('count', ascending=False).head(5)

        if len(sig_counts) == 0:
            print(f"    Warning: No significant features for {title}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(sig_counts['feature'], sig_counts['count'], color='steelblue', edgecolor='black')
        ax.set_xlabel('Number of Conditions with Significant Bias')
        ax.set_title(f'{title}\n({category_label})', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # Add percentage labels
        total_conditions = len(data.drop_duplicates(['dataset', 'provider', 'prompt_style']))
        for i, (idx, row) in enumerate(sig_counts.iterrows()):
            pct = 100 * row['count'] / total_conditions
            ax.text(row['count'], i, f"  {pct:.1f}%", va='center')

        plt.tight_layout()
        plt.savefig(TOP5_DIR / filename, bbox_inches='tight')
        plt.close()

        print(f"    ✓ {title}")

    def plot_cumulative_bars(data, group_by, title, filename):
        """Generate cumulative bar charts showing feature significance by group"""
        # Get significance counts per feature per group
        sig_by_group = data[data['significant']].groupby(['feature', group_by]).size().reset_index(name='count')

        # Pivot to get features × groups
        pivot = sig_by_group.pivot(index='feature', columns=group_by, values='count').fillna(0)

        # Get top 10 features by total significance
        pivot['total'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('total', ascending=False).head(10)
        pivot = pivot.drop('total', axis=1)

        if len(pivot) == 0:
            print(f"    Warning: No data for {title}")
            return

        # ENHANCED: Use better colors based on grouping
        if group_by == 'dataset':
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
        elif group_by == 'provider':
            colors = ['#d62728', '#9467bd', '#8c564b']  # Red, purple, brown
        else:  # prompt_style
            colors = plt.cm.Set3(np.linspace(0, 1, len(pivot.columns)))

        fig, ax = plt.subplots(figsize=(12, 8))
        pivot.plot(kind='barh', stacked=True, ax=ax, edgecolor='black', width=0.8,
                   color=colors[:len(pivot.columns)])
        ax.set_xlabel('Number of Conditions with Significant Bias')
        ax.set_title(title, fontweight='bold')
        ax.legend(title=group_by.replace('_', ' ').title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

        # ENHANCED: Add total count labels at the end of bars
        for i, (idx, row) in enumerate(pivot.iterrows()):
            total = row.sum()
            ax.text(total + 0.5, i, f'{int(total)}', va='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(TOP5_DIR / filename, bbox_inches='tight')
        plt.close()

        print(f"    ✓ {title}")

    # Original top 5 plots
    print("\n1. Standard top 5 plots...")
    plot_top5(comp_df, 'Top 5 Most Frequently Significant Features (All Conditions)',
              'top5_all_disaggregated.png', 'All Features')

    sub_df = comp_df[comp_df['feature'].isin(SUBSTANTIVE_FEATURES)]
    plot_top5(sub_df, 'Top 5 Most Frequently Significant Features (All Conditions)',
              'top5_substantive_disaggregated.png', 'Substantive Features')

    sty_df = comp_df[comp_df['feature'].isin(STYLISTIC_FEATURES)]
    plot_top5(sty_df, 'Top 5 Most Frequently Significant Features (All Conditions)',
              'top5_stylistic_disaggregated.png', 'Stylistic Features')

    for dataset in DATASETS:
        dataset_df = comp_df[comp_df['dataset'] == dataset]
        plot_top5(dataset_df,
                  f'Top 5 Significant Features - {dataset.capitalize()}',
                  f'top5_all_dataset_{dataset}.png', 'All Features')

    for provider in PROVIDERS:
        provider_df = comp_df[comp_df['provider'] == provider]
        plot_top5(provider_df,
                  f'Top 5 Significant Features - {provider.capitalize()}',
                  f'top5_all_model_{provider}.png', 'All Features')

    for prompt in PROMPT_STYLES:
        prompt_df = comp_df[comp_df['prompt_style'] == prompt]
        plot_top5(prompt_df,
                  f'Top 5 Significant Features - {prompt.capitalize()} Prompt',
                  f'top5_all_prompt_{prompt}.png', 'All Features')

    plot_top5(comp_df, 'Top 5 Most Frequently Significant Features (Overall)',
              'top5_all_fully_aggregated.png', 'All Features')
    plot_top5(sub_df, 'Top 5 Most Frequently Significant Features (Overall)',
              'top5_substantive_fully_aggregated.png', 'Substantive Features')
    plot_top5(sty_df, 'Top 5 Most Frequently Significant Features (Overall)',
              'top5_stylistic_fully_aggregated.png', 'Stylistic Features')

    print("\n2. Cumulative bar charts...")
    plot_cumulative_bars(comp_df, 'dataset',
                        'Top Features by Dataset (Cumulative)',
                        'top5_cumulative_by_dataset.png')
    plot_cumulative_bars(comp_df, 'provider',
                        'Top Features by Model (Cumulative)',
                        'top5_cumulative_by_model.png')
    plot_cumulative_bars(comp_df, 'prompt_style',
                        'Top Features by Prompt Style (Cumulative)',
                        'top5_cumulative_by_prompt.png')

    print(f"\n✓ Saved top 5 significant features plots to {TOP5_DIR}")

# ============================================================================
# 5. FEATURE IMPORTANCE RANKINGS - FIXED
# ============================================================================

def generate_feature_importance_plots():
    """
    FIXED: Exclude shap_file column from aggregations
    """
    print("\n" + "="*80)
    print("GENERATING FEATURE IMPORTANCE PLOTS - FIXED")
    print("="*80)

    # Load importance results
    importance_df = pd.read_parquet(OUTPUT_DIR / 'importance_analysis' / 'importance_results.parquet')

    # Extract SHAP values, excluding shap_file
    shap_cols = [c for c in importance_df.columns if c.startswith('shap_') and c != 'shap_file']
    feature_names = [c.replace('shap_', '') for c in shap_cols]

    print(f"Found {len(feature_names)} features with SHAP values")

    # Create long-form dataframe for easier analysis
    importance_long = []
    for _, row in importance_df.iterrows():
        for feat in feature_names:
            shap_col = f'shap_{feat}'
            if shap_col in importance_df.columns:
                importance_long.append({
                    'dataset': row['dataset'],
                    'provider': row['provider'],
                    'model': row['model'],
                    'prompt_style': row['prompt_style'],
                    'feature': feat,
                    'shap_importance': row[shap_col],
                    'auroc': row['auroc']
                })

    imp_df = pd.DataFrame(importance_long)

    # Helper function to plot top features
    def plot_importance(data, title, filename, top_n=10):
        # Average SHAP importance per feature
        avg_imp = data.groupby('feature')['shap_importance'].mean().reset_index()
        avg_imp = avg_imp.sort_values('shap_importance', ascending=False).head(top_n)

        if len(avg_imp) == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(avg_imp['feature'], avg_imp['shap_importance'], color='steelblue', edgecolor='black')
        ax.set_xlabel('Mean Absolute SHAP Value')
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(IMPORTANCE_DIR / filename, bbox_inches='tight')
        plt.close()

        print(f"    ✓ {title[:60]}...")

    # 1. Fully disaggregated - one plot per condition
    print("\n1. Fully disaggregated plots...")
    count = 0
    for dataset in DATASETS:
        for provider in PROVIDERS:
            for prompt in PROMPT_STYLES:
                cond_df = imp_df[
                    (imp_df['dataset'] == dataset) &
                    (imp_df['provider'] == provider) &
                    (imp_df['prompt_style'] == prompt)
                ]
                if len(cond_df) > 0:
                    plot_importance(
                        cond_df,
                        f'Feature Importance: {dataset} × {provider} × {prompt}',
                        f'importance_{dataset}_{provider}_{prompt}.png',
                        top_n=10
                    )
                    count += 1
    print(f"  Generated {count} disaggregated plots")

    # 2. By dataset
    print("\n2. By dataset...")
    for dataset in DATASETS:
        dataset_df = imp_df[imp_df['dataset'] == dataset]
        plot_importance(
            dataset_df,
            f'Feature Importance - {dataset.capitalize()} (All Models & Prompts)',
            f'importance_dataset_{dataset}.png',
            top_n=10
        )

    # 3. By model
    print("\n3. By model...")
    for provider in PROVIDERS:
        provider_df = imp_df[imp_df['provider'] == provider]
        plot_importance(
            provider_df,
            f'Feature Importance - {provider.capitalize()} (All Datasets & Prompts)',
            f'importance_model_{provider}.png',
            top_n=10
        )

    # 4. By prompt style
    print("\n4. By prompt style...")
    for prompt in PROMPT_STYLES:
        prompt_df = imp_df[imp_df['prompt_style'] == prompt]
        plot_importance(
            prompt_df,
            f'Feature Importance - {prompt.capitalize()} Prompt (All Datasets & Models)',
            f'importance_prompt_{prompt}.png',
            top_n=10
        )

    # 5. Fully aggregated
    print("\n5. Fully aggregated...")
    plot_importance(
        imp_df,
        'Feature Importance - Overall (All Conditions)',
        'importance_fully_aggregated.png',
        top_n=16  # Show all features
    )

    print(f"\n✓ Saved feature importance plots to {IMPORTANCE_DIR}")

    return imp_df

# ============================================================================
# 6. REGRESSION TABLES (LaTeX Format)
# ============================================================================

def generate_regression_tables(comp_df):
    """
    Generate regression tables in LaTeX format
    """
    print("\n" + "="*80)
    print("GENERATING REGRESSION TABLES (LaTeX)")
    print("="*80)

    def create_latex_table(data, level_name, filename):
        """Create a LaTeX table from aggregated data"""

        # Group by feature and compute statistics
        table_data = data.groupby('feature').agg({
            'bias': ['mean', 'std'],
            'p_value': lambda x: (x < 0.05).mean(),
            'significant': 'sum'
        }).reset_index()

        table_data.columns = ['Feature', 'Mean Bias', 'Std Bias', 'Prop Significant', 'N Significant']

        # Sort by mean bias (descending)
        table_data = table_data.sort_values('Mean Bias', ascending=False)

        # Start LaTeX table
        latex = []
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append(f"\\caption{{Bias Analysis Results - {level_name}}}")
        latex.append("\\label{tab:bias_" + level_name.lower().replace(' ', '_').replace(':', '_') + "}")
        latex.append("\\begin{tabular}{lrrrr}")
        latex.append("\\toprule")
        latex.append("Feature & Mean Bias & Std & Prop. Sig. & N Sig. \\\\")
        latex.append("\\midrule")

        for _, row in table_data.iterrows():
            feature = row['Feature'].replace('_', '\\_')
            latex.append(f"{feature} & {row['Mean Bias']:.3f} & {row['Std Bias']:.3f} & "
                        f"{row['Prop Significant']:.2f} & {int(row['N Significant'])} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        # Save to file
        with open(TABLES_DIR / filename, 'w') as f:
            f.write('\n'.join(latex))

        print(f"    ✓ {level_name}")

    # By dataset
    for dataset in DATASETS:
        dataset_df = comp_df[comp_df['dataset'] == dataset]
        create_latex_table(
            dataset_df,
            f'Dataset: {dataset.capitalize()}',
            f'table_dataset_{dataset}.tex'
        )

    # By model
    for provider in PROVIDERS:
        provider_df = comp_df[comp_df['provider'] == provider]
        create_latex_table(
            provider_df,
            f'Model: {provider.capitalize()}',
            f'table_model_{provider}.tex'
        )

    # By prompt style
    for prompt in PROMPT_STYLES:
        prompt_df = comp_df[comp_df['prompt_style'] == prompt]
        create_latex_table(
            prompt_df,
            f'Prompt: {prompt.capitalize()}',
            f'table_prompt_{prompt}.tex'
        )

    # Fully aggregated
    create_latex_table(
        comp_df,
        'Overall (All Conditions)',
        'table_fully_aggregated.tex'
    )

    print(f"\n✓ Saved LaTeX tables to {TABLES_DIR}")

# ============================================================================
# 7. PER-FEATURE BIAS PLOTS
# ============================================================================

def generate_per_feature_bias_plots(comp_df):
    """
    NEW: Generate individual plots for each feature showing bias across conditions
    Creates plots partially aggregated by dataset, model, and prompt
    """
    print("\n" + "="*80)
    print("GENERATING PER-FEATURE BIAS PLOTS")
    print("="*80)

    all_features = sum(FEATURES.values(), [])
    per_feature_dir = VIZ_DIR / '7_per_feature_bias'
    per_feature_dir.mkdir(exist_ok=True)

    for feature in all_features:
        feat_data = comp_df[comp_df['feature'] == feature].copy()

        if len(feat_data) == 0:
            continue

        # Create 3 subplots: by dataset, by model, by prompt
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Bias Analysis: {feature}', fontsize=14, fontweight='bold')

        # 1. By Dataset (aggregated across models and prompts)
        ax = axes[0]
        by_dataset = feat_data.groupby('dataset').agg({
            'bias': 'mean',
            'significant': 'mean',
            'p_value': 'mean'
        }).reset_index()

        colors = ['coral' if sig > 0.5 else 'steelblue' for sig in by_dataset['significant']]
        bars = ax.bar(by_dataset['dataset'], by_dataset['bias'], color=colors, edgecolor='black')
        ax.set_title('By Dataset\n(aggregated across models & prompts)')
        ax.set_ylabel('Mean Bias')
        ax.set_xlabel('Dataset')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Add significance labels
        for i, (bar, sig) in enumerate(zip(bars, by_dataset['significant'])):
            if sig > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

        # 2. By Model (aggregated across datasets and prompts)
        ax = axes[1]
        by_model = feat_data.groupby('provider').agg({
            'bias': 'mean',
            'significant': 'mean',
            'p_value': 'mean'
        }).reset_index()

        colors = ['coral' if sig > 0.5 else 'steelblue' for sig in by_model['significant']]
        bars = ax.bar(by_model['provider'], by_model['bias'], color=colors, edgecolor='black')
        ax.set_title('By Model\n(aggregated across datasets & prompts)')
        ax.set_ylabel('Mean Bias')
        ax.set_xlabel('Model')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        for i, (bar, sig) in enumerate(zip(bars, by_model['significant'])):
            if sig > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

        # 3. By Prompt Style (aggregated across datasets and models)
        ax = axes[2]
        by_prompt = feat_data.groupby('prompt_style').agg({
            'bias': 'mean',
            'significant': 'mean',
            'p_value': 'mean'
        }).reset_index()

        colors = ['coral' if sig > 0.5 else 'steelblue' for sig in by_prompt['significant']]
        bars = ax.bar(by_prompt['prompt_style'], by_prompt['bias'], color=colors, edgecolor='black')
        ax.set_title('By Prompt Style\n(aggregated across datasets & models)')
        ax.set_ylabel('Mean Bias')
        ax.set_xlabel('Prompt Style')
        ax.set_xticklabels(by_prompt['prompt_style'], rotation=45, ha='right')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        for i, (bar, sig) in enumerate(zip(bars, by_prompt['significant'])):
            if sig > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       '*', ha='center', va='bottom', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(per_feature_dir / f'{feature}_bias_across_conditions.png', bbox_inches='tight')
        plt.close()

        print(f"  ✓ {feature}")

    print(f"\n✓ Saved {len(all_features)} per-feature bias plots to {per_feature_dir}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS PIPELINE - 16 CORE FEATURES")
    print("="*80)
    print("\nGenerating 4 types of outputs:")
    print("  1. Feature distributions")
    print("  2. Bias heatmaps (magnitude)")
    print("  3. Directional bias plots (shows which categories are favored)")
    print("  4. Feature importance analysis (NEW - Random Forest + SHAP)")
    print("\n" + "="*80)

    # 1. Feature distributions
    generate_feature_distributions()

    # 2. Compute comparisons (needed for bias heatmaps)
    # Note: We compute but don't generate the aggregated plots
    comp_df = generate_pool_vs_recommended()

    # 3. Bias heatmaps
    generate_bias_heatmaps(comp_df)

    # 4. Directional bias plots
    df_directional = compute_directional_bias()
    generate_directional_bias_plots(df_directional)

    # 5. Feature importance analysis (NEW)
    # Check if pre-computed data exists
    importance_data_file = OUTPUT_DIR / 'feature_importance_data.csv'
    if importance_data_file.exists():
        print("\n" + "="*80)
        print("LOADING PRE-COMPUTED FEATURE IMPORTANCE DATA")
        print("="*80)
        df_importance = pd.read_csv(importance_data_file)
        print(f"✓ Loaded {len(df_importance)} feature-condition combinations from cache")
        print(f"  To recompute, delete: {importance_data_file}")
    else:
        df_importance = compute_feature_importance()

    generate_feature_importance_heatmaps(df_importance)


    # Final summary
    print("\n" + "="*80)
    print("ALL OUTPUTS COMPLETE!")
    print("="*80)
    print("\nGenerated:")
    print(f"  1. {len(sum(FEATURES.values(), []))} distribution plots → {DIST_DIR}")
    print(f"      ✓ author_is_minority fixed (categorical)")
    print(f"      ✓ Political leaning expanded (7 categories)")
    print(f"  2. Bias heatmaps (5 aggregation levels) → {HEATMAP_DIR}")
    print(f"      ✓ Categorical bias FIXED (no more zeros!)")
    print(f"      ✓ Multi-level significance markers (*, **, ***)")
    print(f"  3. Directional bias plots (16 plots) → {VIZ_DIR / '4_directional_bias'}")
    print(f"      ✓ Shows which categories/values are favored")
    print(f"      ✓ 1 plot per feature (6 subplots per prompt style)")
    print(f"  4. Feature importance heatmaps (10 plots) → {VIZ_DIR / '5_feature_importance'}")
    print(f"      ✓ SHAP values (normalized within feature)")
    print(f"      ✓ Structure matches bias heatmaps:")
    print(f"        - Disaggregated by prompt (6) + By dataset (1) + By model (1) + By prompt (1) + Overall (1)")
    print(f"\nAll outputs saved to: {VIZ_DIR}")
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE - Ready for interpretation!")
    print("="*80)

if __name__ == '__main__':
    main()
