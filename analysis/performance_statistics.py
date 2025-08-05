import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional

# Edit this list to change which phenomena are included in the table
PHENOMENA_LIST = [
    "adjunct_island",
    "anaphor_gender_agreement",
    "anaphor_number_agreement",
    "animate_subject_passive",
    "distractor_agreement_relative_clause",
    "ellipsis_n_bar_1",
    "irregular_past_participle_adjectives",
    "principle_A_c_command",
    "wh_questions_object_gap",
    "wh_questions_object_gap_long_distance",
    "wh_questions_subject_gap",
    "wh_questions_subject_gap_long_distance",
    "left_branch_island_simple_question",
    "determiner_noun_agreement_with_adj_2",
    "existential_there_quantifiers_1"
]

MODEL_ORDER = [
    "english",
    "reverse_full",
    "reverse_partial",
    "shuffle_local3",
    "shuffle_local5",
    "shuffle_local10",
    "shuffle_even_odd",
    "shuffle_deterministic21",
    "shuffle_nondeterministic",
]

def save_performance_table_image(
    phenomena: List[str] = PHENOMENA_LIST,
    csv_path: str = "experiments/output/v2/results.csv",
    output_dir: str = "analysis/output",
    output_filename: str = "performance_table.png"
):
    """
    Generates and saves a PNG table of model performance by grammatical phenomenon.
    Args:
        phenomena: List of grammatical phenomena to include (default: all in CSV)
        csv_path: Path to results CSV
        output_dir: Directory to save PNG
        output_filename: Name of PNG file
    """
    df = pd.read_csv(csv_path)
    # Extract all unique phenomena and models
    all_models = MODEL_ORDER
    # Filter to requested phenomena
    df = df[df["grammatical phenomenon"].isin(phenomena)]

    # Pivot table: index=phenomenon, columns=model, values=accuracy
    table = df.pivot_table(index="grammatical phenomenon", columns="model name", values="accuracy")
    # Ensure all models/phenomena present (fill missing with NaN)
    table = table.reindex(index=phenomena, columns=all_models)

    # Add 'Overall' row (average per model)
    table.loc['Overall'] = table.mean(axis=0, skipna=True)

    # Format for display
    display_table = table.round(3)

    # Plot as heatmap
    plt.figure(figsize=(max(2 + len(all_models) * 1.1, 8), max(1 + len(phenomena) * 0.5, 8)))
    ax = sns.heatmap(display_table, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, linewidths=0.5, linecolor='gray',
                     annot_kws={"size": 10})
    ax.set_xlabel("Model")
    ax.set_ylabel("Grammatical Phenomenon")
    # ax.set_title("Model Accuracy by Grammatical Phenomenon")
    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, output_filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Performance table saved to {outpath}")

def percent_non_english_best_model(csv_path="experiments/output/v2/results.csv"):
    """
    Calculates the percentage of grammatical phenomena where the highest performing model was NOT 'english'.
    Also prints a table with, for every task, the accuracy of english and the accuracy/model of the best model.
    """
    df = pd.read_csv(csv_path)
    # Get the row with the highest accuracy for each grammatical phenomenon
    idx = df.groupby("grammatical phenomenon")["accuracy"].idxmax()
    best_per_phenomenon = df.loc[idx]

    # Get the english model for each phenomenon
    english_per_phenomenon = (
        df[df["model name"] == "english"]
        .set_index("grammatical phenomenon")
        .loc[best_per_phenomenon["grammatical phenomenon"]]
    )

    # Print the table
    print(f"{'Phenomenon':35} | {'English Acc':>11} | {'Best Model':>15} | {'Best Acc':>8}")
    print("-" * 80)
    for _, row in best_per_phenomenon.iterrows():
        phenomenon = row["grammatical phenomenon"]
        best_model = row["model name"]
        best_acc = row["accuracy"]
        english_acc = english_per_phenomenon.loc[phenomenon, "accuracy"]
        print(f"{phenomenon:35} | {english_acc:11.3f} | {best_model:15} | {best_acc:8.3f}")

    # Count how many times the best model is not 'english'
    non_english_count = (best_per_phenomenon["model name"] != "english").sum()
    total = len(best_per_phenomenon)
    percent = (non_english_count / total) * 100 if total > 0 else 0
    print(f"\nOut of {total} phenomena, {non_english_count} ({percent:.1f}%) had a best model that was NOT 'english'.")
    return percent

def main():
    save_performance_table_image(PHENOMENA_LIST, output_filename="performance_table_overall.png")
    save_performance_table_image([
            "anaphor_gender_agreement",
            "anaphor_number_agreement",
            "animate_subject_passive",
            "determiner_noun_agreement_with_adj_2",
            "existential_there_quantifiers_1"
        ], output_filename="performance_table_local.png")
    save_performance_table_image([
        "adjunct_island",
        "distractor_agreement_relative_clause",
        "ellipsis_n_bar_1",
        "principle_A_c_command",
        "wh_questions_object_gap",
        "wh_questions_object_gap_long_distance",
    ], output_filename="performance_table_structural.png")
    # percent_non_english_best_model()

if __name__ == "__main__":
    main()