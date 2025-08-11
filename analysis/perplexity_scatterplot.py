import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_vs_perplexity_good(model_names):
    """
    Plots accuracy vs perplexity-good for the given list of model names.
    Computes and displays the Pearson correlation coefficient.
    Adds regression lines to the scatter plot.
    Saves the plot as a PNG in analysis/output/perplexity.
    """
    import os
    from scipy.stats import pearsonr
    df = pd.read_csv('experiments/output/v2/results.csv')
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))

    # Ensure output directory exists
    output_dir = 'analysis/output/perplexity'
    os.makedirs(output_dir, exist_ok=True)

    colors = plt.cm.get_cmap('tab10', len(model_names))
    for idx, model_name in enumerate(model_names):
        model_df = df[df['model name'] == model_name]
        if model_df.empty:
            print(f"No data found for model: {model_name}")
            continue
        x = model_df['perplexity good']
        y = model_df['accuracy']
        ax.scatter(
            x,
            y,
            label=f"{model_name}",
            s=120,
            alpha=0.7,
            color=colors(idx)
        )
        # Fit regression line
        sns.regplot(
            x=x,
            y=y,
            scatter=False,
            ax=ax,
            color=colors(idx),
            line_kws={'label':f"{model_name} fit"}
        )
        # Compute Pearson correlation
        if len(x) > 1:
            corr, pval = pearsonr(x, y)
            corr_text = f"r = {corr:.2f} (p={pval:.2g})"
            # Place annotation near the top right of the model's data
            max_x = x.max()
            max_y = y.max()
            ax.annotate(f"{model_name}: {corr_text}",
                        xy=(max_x, max_y),
                        xycoords='data',
                        xytext=(10, 0),
                        textcoords='offset points',
                        fontsize=11,
                        color=colors(idx),
                        weight='bold',
                        va='top',
                        bbox=dict(facecolor='white', edgecolor=colors(idx), boxstyle='round,pad=0.2', alpha=0.7))
            print(f"{model_name}: Pearson r = {corr:.3f}, p = {pval:.3g}")
        else:
            print(f"{model_name}: Not enough data for correlation.")

    ax.set_xlabel('Perplexity (Good Sentences)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Accuracy vs Perplexity (Good Sentences)', fontsize=16)
    ax.legend(title='Model Name')
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # Create an appropriate filename
    if len(model_names) == 1:
        fname = f"accuracy_vs_perplexity_good_{model_names[0]}.png"
    else:
        fname = "accuracy_vs_perplexity_good_" + "_".join(model_names) + ".png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot to {save_path}")

# Existing function preserved for reference

def plot_accuracy_vs_perplexity_diff():
    # Load the dataset
    df = pd.read_csv('experiments/output/v2/results.csv')

    # Calculate the difference in perplexity between ungrammatical ("bad") and grammatical ("good") sentences.
    df['perplexity_diff'] = df['perplexity bad'] - df['perplexity good']

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = sns.scatterplot(
        data=df,
        x='perplexity_diff',
        y='accuracy',
        hue='grammatical phenomenon',
        style='model name',
        s=150,
        alpha=0.8,
        ax=ax
    )

    ax.set_xlabel('Perplexity Difference (Bad Sentence - Good Sentence)', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.axhline(0.5, ls='--', color='gray', lw=1.5)
    ax.axvline(0, ls='--', color='gray', lw=1.5)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig('analysis/output/accuracy_vs_perplexity_analysis.png')
    print("Scatter plot 'accuracy_vs_perplexity_analysis.png' has been generated.")


def main():
    # plot_accuracy_vs_perplexity_diff()

    # Plot accuracy vs perplexity-good for each model individually
    model_names = [
        'english',
        'shuffle_deterministic21',
        'shuffle_local3',
        'shuffle_local5',
        'shuffle_local10',
        'shuffle_even_odd',
        'reverse_partial',
        'reverse_full',
        'shuffle_nondeterministic',
    ]
    for model in model_names:
        plot_accuracy_vs_perplexity_good([model])

if __name__ == '__main__':
    main()
