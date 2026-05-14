import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import initialize_style


def plot_perplexity_boxplot(score_path):
    # Merge the two dataframes on 'model name'
    df = pd.read_csv(score_path)
    palette = initialize_style()

    df["gen_length"] = df["generation"].apply(lambda x: len(x.split()))

    df = df[(df["ntokens"] >= 10) & (df["ntokens"] <= 20)]

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

    # make aspect ratio wider:
    plt.figure(figsize=(5, 25 / 8))

    ax = sns.boxplot(
        x="perturbation",
        y="perplexity",
        hue="perturbation",
        data=df,
        showfliers=False,
        order=column_order,
        hue_order=column_order,
        legend=False,
        palette=palette,
    )
    ax.set_box_aspect(6 / 8)

    # set xticks text manually
    plt.xticks(
        ticks=range(len(column_order)),
        labels=["E", "FR", "PR", "S3", "S5", "SEO", "S10", "DS", "NDS"],
    )
    ax.yaxis.label.set_fontsize(12)

    plt.xlabel("")
    plt.ylabel("GPT2-large perplexity")
    plt.savefig("output/figures/perplexity_boxplot.pdf", bbox_inches="tight")


def boxplot_stats(score_path, mlocal_path, perplexity_path, out_path):
    df = pd.read_csv(score_path)
    perp = pd.read_csv(perplexity_path)
    mlocal = pd.read_csv(mlocal_path)

    df["gen_length"] = df["generation"].apply(lambda x: len(x.split()))

    df = df[(df["ntokens"] >= 10) & (df["ntokens"] <= 20)]

    # list of medians for each perturbation, with the perturbation in the dataframe:
    medians = df.groupby("perturbation")["perplexity"].median()

    # correlate means with perplexity and mlocal:
    from scipy.stats import spearmanr

    # merge perp and mlocal on model name
    merged = pd.merge(perp, medians, left_on="model name", right_on="perturbation")
    correlation_perp, p_value_perp = spearmanr(
        merged["perplexity_x"], merged["perplexity_y"]
    )

    merged_mlocal = pd.merge(
        mlocal, medians, left_on="model name", right_on="perturbation"
    )
    correlation_mlocal, p_value_mlocal = spearmanr(
        merged_mlocal["perplexity"], merged_mlocal["mlocal_4"]
    )

    # write output csv of correlations
    out_df = pd.DataFrame(
        {
            "metric": ["perplexity", "mlocal_4"],
            "spearman_rho": [correlation_perp, correlation_mlocal],
            "p_value": [p_value_perp, p_value_mlocal],
        }
    )

    out_df.to_csv(out_path, index=False)
