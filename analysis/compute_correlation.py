import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from analysis.performance_statistics import (
    get_phenomena_by_cue_reliability,
    GRAMMATICAL_PHENOMENA_TABLE_CSV_PATH,
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
    analyze_ranking_significance(RESULTS_CSV_PATH)
    #main()