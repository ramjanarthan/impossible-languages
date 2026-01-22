import argparse
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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


def spearman_r_from_scores(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, float, int, List[str]]:
    common = sorted(set(a.keys()) & set(b.keys()))
    if len(common) < 2:
        return float("nan"), float("nan"), len(common), common
    a_vals = [a[k] for k in common]
    b_vals = [b[k] for k in common]
    # scipy handles ties and returns both correlation and p-value
    r, p = spearmanr(a_vals, b_vals)
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


def main():
    parser = argparse.ArgumentParser()
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
        df = compute_per_phenomenon_spearman(args.csv, phenomena=phenomena, skip_models=skip_models)
        if args.output:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            df.to_csv(args.output, index=False)
        else:
            print(df.to_string(index=False))
    else:
        r, p, n, common = compute_overall_spearman(args.csv, phenomena=phenomena, skip_models=skip_models)
        print(f"Spearman r (overall): {r:.4f} , p (overall): {p:.4f} using {n} common models")
        print(f"Common models: {', '.join(common)}")


if __name__ == "__main__":
    main()