import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from utils import initialize_style

save_df = []


def plot_metric_proportion(
    df, metric="slor", bin_size=5, ci_method="wilson", out_path=None
):
    """
    Plot proportion of sentences below the 75th percentile of English for a given metric,
    binned by number of tokens.

    Parameters:
    - df: DataFrame with columns ['ntokens', 'perturbation', metric]
    - metric: metric to use ('slor', 'perplexity', 'morcela', etc.)
    - bin_size: size of the ntokens bins
    - ci_method: method for confidence intervals ('wilson', 'normal', 'jeffreys', etc.)
    """
    palette = initialize_style()

    column_order = [
        "english",
        "reverse_full",
        "reverse_partial",
        "shuffle_local3",
        "shuffle_local5",
        "shuffle_even_odd",
        "shuffle_local10",
        "shuffle_deterministic21",
        "shuffle_nondeterministic",
    ]
    # Create bins dynamically
    max_tokens = df["ntokens"].max()
    bins = list(range(0, max_tokens + bin_size, bin_size))
    df["ntokens_bucket"] = pd.cut(df["ntokens"], bins=bins)

    # Compute English quantiles for the metric in each bucket
    english_quantiles = (
        df[df["perturbation"] == "english"]
        .groupby("ntokens_bucket")[metric]
        .quantile([0.25, 0.5, 0.75])
        .unstack(level=-1)
    )
    english_quantiles.columns = [f"{metric}_q25", f"{metric}_q50", f"{metric}_q75"]

    # Merge back to df
    df = df.merge(english_quantiles, on="ntokens_bucket", how="left")

    # Flag sentences below 75th percentile
    df[f"below_{metric}_q75"] = df[metric] < df[f"{metric}_q75"]

    # # print 5 examples below and above the 75th percentile for each perturbation, in bucks (5, 10] and (20, 25])
    # for perturbation in df["perturbation"].unique():
    #     for bucket in (pd.Interval(10, 15),):
    #         sub_df = df[
    #             (df["perturbation"] == perturbation) & (df["ntokens_bucket"] == bucket)
    #         ]
    #         if not sub_df.empty:
    #             print(f"Perturbation: {perturbation}, Bucket: {bucket}")
    #             print("Below 75th percentile:")
    #             print(
    #                 sub_df[sub_df[f"below_{metric}_q75"]]
    #                 .sort_values(by=metric, ascending=True)["generation"]
    #                 .head(5)
    #                 .to_list()
    #             )
    #             print("Above 75th percentile:")
    #             print(
    #                 sub_df[~sub_df[f"below_{metric}_q75"]]
    #                 .sort_values(by=metric, ascending=False)["generation"]
    #                 .head(5)
    #                 .to_list()
    #             )
    #             print("\n")

    # Group for plotting
    plot_df = (
        df.dropna(subset=["ntokens_bucket"])
        .groupby(["perturbation", "ntokens_bucket"])
        .agg(n=("ntokens", "size"), prop_below=(f"below_{metric}_q75", "mean"))
        .reset_index()
    )

    # Calculate confidence intervals
    ci_low, ci_upp = proportion_confint(
        count=plot_df["prop_below"] * plot_df["n"], nobs=plot_df["n"], method=ci_method
    )
    plot_df["ci_lower"] = ci_low
    plot_df["ci_upper"] = ci_upp

    # Position for plotting
    plot_df["plot_position"] = plot_df["ntokens_bucket"].apply(
        lambda x: x.left + bin_size / 2
    )

    # Plot
    sns.set_style("whitegrid")
    plt.rcParams["font.family"] = "Futura"
    # turn off borders
    plt.figure(figsize=(5, 5 * (5 / 8)))
    ax = sns.lineplot(
        x="plot_position",
        y="prop_below",
        hue="perturbation",
        hue_order=column_order,
        palette=palette,
        data=plot_df,
        marker="o",
    )

    # Add confidence intervals
    for perturbation in plot_df["perturbation"].unique():
        sub_df = plot_df[plot_df["perturbation"] == perturbation]
        ax.fill_between(
            sub_df["plot_position"],
            sub_df["ci_lower"],
            sub_df["ci_upper"],
            alpha=0.1,
            color="#AAAAAA",
        )

    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.9, pos.height])
    ax.set_xlim(0, 50)
    # Customize legend labels
    handles, labels = ax.get_legend_handles_labels()

    # make the handles into dots instead of lines
    for handle in handles:
        handle.set_linestyle("")
        handle.set_marker("o")
        handle.set_markersize(8)

    label_mapping = {
        "english": "E",
        "reverse_full": "FR",
        "reverse_partial": "PR",
        "shuffle_local3": "S3",
        "shuffle_local5": "S5",
        "shuffle_even_odd": "SEO",
        "shuffle_local10": "S10",
        "shuffle_deterministic21": "DS",
        "shuffle_nondeterministic": "NDS",
    }
    new_labels = [label_mapping.get(label, label) for label in labels]
    ax.legend(
        handles,
        new_labels,
        loc="upper center",
        bbox_to_anchor=(0.47, -0.2),
        ncol=9,
        frameon=False,
        labelspacing=0.1,
        handletextpad=0.2,
        columnspacing=0.7,
    )
    ax.set_box_aspect(6 / 8)

    ax.set_xlabel("# generated tokens")
    ax.set_ylabel(f"Prop. < 75% English {metric}")
    # make fonts bigger
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

    plt.savefig(
        f"output/figures/proportion_below_english_{metric}.pdf", bbox_inches="tight"
    )

    return plot_df


# Example usage:
def plot_quantiles(score_path):
    df = pd.read_csv(score_path)
    plot_df = plot_metric_proportion(
        df,
        metric="perplexity",
        bin_size=5,
        ci_method="wilson",
    )
    plot_df.to_csv(
        "output/figures/proportion_below_english_perplexity.csv", index=False
    )
