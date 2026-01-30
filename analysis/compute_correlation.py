import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from analysis.performance_statistics import (
    get_phenomena_by_cue_reliability,
    GRAMMATICAL_PHENOMENA_TABLE_CSV_PATH,
    MODEL_ORDER,
    MODEL_TO_DISPLAY_NAME
)

RESULTS_CSV_PATH = "experiments/output/v3/results.csv"

def load_results(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def get_accuracy_by_model(
    df: pd.DataFrame,
    phenomena: Optional[Iterable[str]] = None,
    skip_models: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    dff = df
    if phenomena is not None:
        phenomena = list(phenomena)
        dff = dff[dff["grammatical phenomenon"].isin(phenomena)]
    if skip_models is not None:
        dff = dff[~dff["model name"].isin(list(skip_models))]
    if dff.empty:
        return {}
    perf = dff.groupby("model name")["accuracy"].mean()
    return perf.to_dict()

def rank_values(values: List[float]) -> np.ndarray:
    s = pd.Series(values)
    return s.rank(method="average").to_numpy()


def friendman_from_scores(df: pd.DataFrame) -> Tuple[float, float]:
    stat, p_friedman = stats.friedmanchisquare(*[df[df["model name"] == model]["accuracy"].values for model in df["model name"].unique()])
    return stat, p_friedman

def spearman_r_from_scores(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, float, int, List[str]]:
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 2:
        return float("nan"), float("nan"), len(common), common
    a_vals = [a[k] for k in common]
    b_vals = [b[k] for k in common]
    # scipy handles ties and returns both correlation and p-value
    r, p = stats.spearmanr(a_vals, b_vals)
    r = float(r)
    p = float(p)
    return r, p, len(common), common

def get_mlocal_scores() -> Dict[str, float]:
    labels = [
        "Base",
        "Reverse",
        "EvenOddShuffle",
        "LocalShuffle(K=3)",
        "LocalShuffle(K=5)",
        "LocalShuffle(K=7)",
        "DeterministicShuffle",
    ]
    values = [2.92, 2.98, 3.76, 3.68, 3.88, 4.06, 4.60]
    label_to_model = {
        "Base": "english",
        "Reverse": "reverse_full",
        "EvenOddShuffle": "shuffle_even_odd",
        "LocalShuffle(K=3)": "shuffle_local3",
        "LocalShuffle(K=5)": "shuffle_local5",
        # No known trained model for K=7 in results; skip if absent
        "DeterministicShuffle": "shuffle_deterministic21",
    }
    out: Dict[str, float] = {}
    for lbl, val in zip(labels, values):
        if lbl in label_to_model:
            out[label_to_model[lbl]] = val
    return out


def compute_overall_spearman(
    csv_path: str,
    phenomena: Optional[List[str]] = None,
    skip_models: Optional[List[str]] = None,
) -> Tuple[float, float, int, List[str]]:
    df = load_results(csv_path)
    acc = get_accuracy_by_model(df, phenomena=phenomena, skip_models=skip_models)
    mlocal = get_mlocal_scores()
    return spearman_r_from_scores(acc, mlocal)

def compute_per_phenomenon_spearman(
    csv_path: str,
    phenomena: Optional[List[str]] = None,
    skip_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = load_results(csv_path)
    if phenomena is None:
        phenomena = sorted(df["grammatical phenomenon"].unique().tolist())
    rows = []
    mlocal = get_mlocal_scores()
    for ph in phenomena:
        acc = get_accuracy_by_model(df, phenomena=[ph], skip_models=skip_models)
        r, p, n, common = spearman_r_from_scores(acc, mlocal)
        rows.append({
            "phenomenon": ph,
            "spearman_r": r,
            "p_value": p,
            "n_common_models": n,
            "common_models": ",".join(common),
        })
    return pd.DataFrame(rows)

def compute_overall_friedman(
    csv_path: str,
) -> Tuple[float, float]:
    df = load_results(csv_path)
    return friendman_from_scores(df)

def compute_per_phenomenon_friedman(
    csv_path: str,
    phenomena: Optional[List[str]] = None,
    skip_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = load_results(csv_path)
    if phenomena is None:
        phenomena = sorted(df["grammatical phenomenon"].unique().tolist())
    rows = []
    for ph in phenomena:
        acc = get_accuracy_by_model(df, phenomena=[ph], skip_models=skip_models)
        stat, p = friendman_from_scores(acc)
        rows.append({
            "phenomenon": ph,
            "friedman_stat": stat,
            "p_value": p,
        })
    return pd.DataFrame(rows)

def analyze_ranking_significance(csv_path):
    # 1. Load Data
    df = pd.read_csv(csv_path)

    # 2. Define the Models and their 'm-local entropy' Ranks
    # Expectation: As m-local entropy increases, accuracy decreases.
    model_ranks = {
        'english': 1,
        'reverse_full': 2,
        'shuffle_local3': 3,
        'reverse_partial': 4,
        'shuffle_even_odd': 5,
        'shuffle_local5': 6,
        'shuffle_deterministic21': 7,
        'shuffle_local10': 8,
        'shuffle_nondeterministic': 9
    }
    
    df_subset = df[df['model name'].isin(model_ranks.keys())].copy()
    
    # 3. Pivot Data (Rows=Datasets, Cols=Models)
    pivot_df = df_subset.pivot_table(
        index='grammatical phenomenon', 
        columns='model name', 
        values='accuracy'
    )
    
    # Ensure columns are sorted by the m-local entropy rank
    sorted_models = sorted(model_ranks.keys(), key=lambda x: model_ranks[x])
    pivot_df = pivot_df[sorted_models]
    
    print(f"Analyzing {len(pivot_df)} datasets across all {len(sorted_models)} models")
    print(f"Model Order (m-local entropy): {sorted_models}")
    
    # 4. Friedman Test (Validation Step)
    # Checks if there is ANY difference between the models
    stat, p_friedman = stats.friedmanchisquare(*[pivot_df[col] for col in pivot_df.columns])
    print(f"\n[Validation] Friedman Test: Chi2 = {stat:.4e},  p = {p_friedman:.4e}")
    if p_friedman < 0.05:
        print(" -> PASSED: Models are statistically different.")
    else:
        print(" -> WARNING: Models may not be distinguishable.")

    # 5. Calculate Kendall's Tau for EACH dataset individually
    taus = []
    m_local_vector = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) # The ideal rank order
    
    for dataset_name, row in pivot_df.iterrows():
        perfs = row.values
        # Kendall's Tau between (Model Performance) and (m-local entropy)
        # We expect a NEGATIVE correlation (Higher Window -> Lower Accuracy)
        tau, p = stats.kendalltau(perfs, m_local_vector)
        if not np.isnan(tau):
            taus.append(tau)

    # 6. Wilcoxon Signed-Rank Test
    # We test if the list of 69 correlation coefficients is significantly < 0
    w_stat, p_wilcoxon = stats.wilcoxon(taus, alternative='less')

    mean_tau = np.mean(taus)
    print("\n Hypothesis: As m-local entropy increases, accuracy decreases. Testing if the list of tau values is significantly < 0")    
    print(f"Mean Kendall's Tau: {mean_tau:.4f} for {len(taus)} datasets")
    print(f"Wilcoxon Signed-Rank Test: p = {p_wilcoxon:.4e}")
    
    if p_wilcoxon < 0.05:
        print(" -> SIGNIFICANT: The ordering of m-local entropy systematically affects model performance.")
    else:
        print(" -> NOT SIGNIFICANT: The relationship is not consistent across datasets.")

def compute_pairwise_wilcoxon(csv_path):
    """
    Compute pairwise Wilcoxon signed-rank tests between all pairs of models.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of (p_values_matrix, statistics_matrix)
            where each is a DataFrame with models as both rows and columns.
    """
    # 1. Load Data
    df = pd.read_csv(csv_path)

    n_models = len(MODEL_ORDER)
    p_values = np.ones((n_models, n_models))
    statistics = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i + 1, n_models):
            first_model = MODEL_ORDER[i]
            second_model = MODEL_ORDER[j]

            first_model_results = df[df['model name'] == first_model]
            second_model_results = df[df['model name'] == second_model]
            
            # ensure that results are sorted by grammatical phenomena
            first_model_results = first_model_results.sort_values(by=['grammatical phenomenon'])['accuracy']
            second_model_results = second_model_results.sort_values(by=['grammatical phenomenon'])['accuracy']

            stat, p_value = stats.wilcoxon(first_model_results, second_model_results)
            
            # Store symmetric values
            p_values[i, j] = p_value
            p_values[j, i] = p_value
            statistics[i, j] = stat
            statistics[j, i] = stat

    # Convert to DataFrames with model names as indices
    display_names = [MODEL_TO_DISPLAY_NAME.get(m, m) for m in MODEL_ORDER]
    p_values_df = pd.DataFrame(p_values, index=display_names, columns=display_names)
    statistics_df = pd.DataFrame(statistics, index=display_names, columns=display_names)
    
    return p_values_df, statistics_df


def visualize_pairwise_wilcoxon(
    csv_path: str = RESULTS_CSV_PATH,
    output_path: str = "analysis/output/pairwise_wilcoxon_heatmap.png"
):
    """
    Compute and visualize pairwise Wilcoxon signed-rank test results as a heatmap.
    
    The heatmap shows p-values between all model pairs, with cells colored by 
    significance level. Cells are annotated with p-values and significance markers:
    - *** p < 0.001
    - ** p < 0.01
    - * p < 0.05
    - ns (not significant) p >= 0.05
    
    Args:
        csv_path: Path to the results CSV file
        output_path: Path where the heatmap image will be saved
    """
    # Compute pairwise Wilcoxon tests
    p_values_df, _ = compute_pairwise_wilcoxon(csv_path)
    
    n_models = len(MODEL_ORDER)
    
    # Create annotation matrix with p-values and significance markers
    annotations = []
    for i in range(n_models):
        row = []
        for j in range(n_models):
            if i == j:
                row.append("-")
            else:
                p = p_values_df.iloc[i, j]
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"
                row.append(f"{p:.3f}\n{sig}")
        annotations.append(row)
    
    annot_df = pd.DataFrame(annotations, 
                            index=p_values_df.index, 
                            columns=p_values_df.columns)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a mask for the diagonal
    mask = np.eye(n_models, dtype=bool)
    
    # Create heatmap - lower p-values (more significant) should be darker
    # Using a reversed colormap so significant results stand out
    sns.heatmap(
        p_values_df,
        annot=annot_df,
        fmt="",
        cmap="RdYlGn",  # Red (low p-value/significant) to Green (high p-value/not significant)
        mask=mask,
        vmin=0,
        vmax=0.1,  # Cap at 0.1 to emphasize significance threshold
        cbar_kws={"label": "p-value"},
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"size": 9},
        ax=ax
    )
    
    # Customize the plot
    ax.set_title("Pairwise Wilcoxon Signed-Rank Test Results\n(p-values)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    # Add legend for significance markers
    legend_text = "Significance: *** p<0.001, ** p<0.01, * p<0.05, ns p≥0.05"
    ax.text(0.5, -0.2, legend_text, transform=ax.transAxes, 
            ha="center", va="top", fontsize=10, style="italic")
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Pairwise Wilcoxon heatmap saved to {output_path}")
    
    return p_values_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", default="spearman", choices=["spearman", "friedman"])
    parser.add_argument("--csv", default=RESULTS_CSV_PATH)
    parser.add_argument("--phenomena", default=None, help="Comma-separated list. If omitted, use all.")
    parser.add_argument(
        "--cue_reliability",
        choices=["Strong", "Weak"],
        default=None,
        help="Select phenomena by cue reliability from the grammatical phenomena table.",
    )
    parser.add_argument(
        "--phenomena_csv",
        default=GRAMMATICAL_PHENOMENA_TABLE_CSV_PATH,
        help="Path to grammatical phenomena table CSV (used with --cue_reliability).",
    )
    parser.add_argument("--by_phenomenon", action="store_true")
    parser.add_argument("--skip_models", default=None, help="Comma-separated list to skip.")
    parser.add_argument("--output", default=None, help="Optional path to save CSV of results.")
    args = parser.parse_args()

    phenomena: Optional[List[str]] = None
    # Build phenomena from explicit list
    if args.phenomena:
        phenomena = [s.strip() for s in args.phenomena.split(",") if s.strip()]
    # Optionally restrict by cue reliability
    if args.cue_reliability is not None:
        cr_list = get_phenomena_by_cue_reliability(args.cue_reliability, csv_path=args.phenomena_csv)
        if phenomena is None:
            phenomena = cr_list
        else:
            # Intersect if both provided
            phenomena = [p for p in phenomena if p in set(cr_list)]
    skip_models = None
    if args.skip_models:
        skip_models = [s.strip() for s in args.skip_models.split(",") if s.strip()]

    if args.by_phenomenon:
        if args.test == "spearman":
            df = compute_per_phenomenon_spearman(args.csv, phenomena=phenomena, skip_models=skip_models)
        elif args.test == "friedman":
            df = compute_per_phenomenon_friedman(args.csv, phenomena=phenomena, skip_models=skip_models)
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            df.to_csv(args.output, index=False)
        else:
            print(df.to_string(index=False))
    else:
        if args.test == "spearman":
            r, p, n, common = compute_overall_spearman(args.csv, phenomena=phenomena, skip_models=skip_models)
            print(f"Spearman r (overall): {r:.4f} , p (overall): {p:.4f} using {n} common models")
            common_models = ", ".join(common)
            print(f"Common models: {common_models}")
        elif args.test == "friedman":
            r, p = compute_overall_friedman(args.csv)
            print(f"Friedman chi-square (overall): {r:.4f} , p (overall): {p:.4f}")


if __name__ == "__main__":
    # analyze_ranking_significance(RESULTS_CSV_PATH)
    visualize_pairwise_wilcoxon(RESULTS_CSV_PATH)
    #main()