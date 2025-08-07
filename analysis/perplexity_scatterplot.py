import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('experiments/output/v2/results.csv')

# Calculate the difference in perplexity between ungrammatical ("bad") and grammatical ("good") sentences.
# A larger positive difference means the model is more "surprised" by the bad sentence, which is a good sign.
df['perplexity_diff'] = df['perplexity bad'] - df['perplexity good']

# For visualization, it can be helpful to cap the perplexity difference to make plots more readable,
# as some values can be extremely high. Let's cap it at a reasonable percentile to handle outliers.
# However, for the initial plot, let's see the full range.

# Create the scatter plot
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 10))

# We will use seaborn to easily color the points by the grammatical phenomenon
scatter = sns.scatterplot(
    data=df,
    x='perplexity_diff',
    y='accuracy',
    hue='grammatical phenomenon',
    style='model name',  # Use style to differentiate models
    s=150,  # size of points
    alpha=0.8,
    ax=ax
)

# ax.set_title('Model Accuracy vs. Perplexity Difference', fontsize=18, pad=20)
ax.set_xlabel('Perplexity Difference (Bad Sentence - Good Sentence)', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)

# Add a horizontal line at 0.5 accuracy for reference (chance level)
ax.axhline(0.5, ls='--', color='gray', lw=1.5)
# Add a vertical line at 0 perplexity difference for reference
ax.axvline(0, ls='--', color='gray', lw=1.5)

# Improve the legend
handles, labels = ax.get_legend_handles_labels()
# Place the legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.savefig('analysis/output/accuracy_vs_perplexity_analysis.png')

# Print a summary to confirm the plot has been created.
print("Scatter plot 'accuracy_vs_perplexity_analysis.png' has been generated.")