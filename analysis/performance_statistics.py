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

PHENOMENA_LIST_MAP = {
    "adjunct_island": "island_effects",
    "anaphor_gender_agreement": "anaphor_agreement",
    "anaphor_number_agreement": "anaphor_agreement",
    "animate_subject_passive": "argument_structure",
    "distractor_agreement_relative_clause": "subject_verb_agreement",
    "ellipsis_n_bar_1": "ellipsis",
    "irregular_past_participle_adjectives": "irregular_forms",
    "principle_A_c_command": "binding", 
    "wh_questions_object_gap": "filler_gap_dependency", 
    "wh_questions_object_gap_long_distance": "filler_gap_dependency",
    "wh_questions_subject_gap": "filler_gap_dependency",
    "wh_questions_subject_gap_long_distance": "filler_gap_dependency",
    "left_branch_island_simple_question": "island_effects",
    "determiner_noun_agreement_with_adj_2": "determiner_noun_agreement",
    "existential_there_quantifiers_1": "quantifiers"
}

PHENOMENA_ABR = {
    "island_effects": "ISLAND",
    "anaphor_agreement": "ANA. AGR",
    "argument_structure": "ARG. STR",
    "subject_verb_agreement": "S-V AGR",
    "ellipsis": "ELLIPSIS",
    "irregular_forms": "IRR",
    "binding": "BINDING",
    "filler_gap_dependency": "FILLER. GAP",
    "determiner_noun_agreement": "D-N AGR",
    "quantifiers": "QUANTIFIERS",
    "npi": "NPI",
    "control_raise": "CTRL. RAIS.",
}

PHENOMENA_ORDER = [
    "anaphor_agreement",
    "argument_structure",
    "binding",
    "control_raise",
    "determiner_noun_agreement",
    "ellipsis",
    "filler_gap_dependency",
    "irregular_forms",
    "island_effects",
    "npi",
    "quantifiers",
    "subject_verb_agreement"
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

def plot_phenomenon_group_counts(
    csv_path: str = "experiments/output/v2/results.csv",
    output_path: str = "analysis/output/phenomenon_group_counts.png"
):
    """
    Plots a PNG image: x-axis is phenomenon groups (PHENOMENA_ORDER, labeled with PHENOMENA_ABR),
    one row is the count of unique phenomena in each group (from results.csv).
    """
    import matplotlib.pyplot as plt
    import os
    df = pd.read_csv(csv_path)
    present_phenomena = set(df["grammatical phenomenon"].unique())
    group_counts = {g: 0 for g in PHENOMENA_ORDER}
    for p, g in PHENOMENA_LIST_MAP.items():
        if p in present_phenomena and g in PHENOMENA_ORDER:
            group_counts[g] += 1
    # Prepare data for plotting
    x_labels = [PHENOMENA_ABR.get(g, g) for g in PHENOMENA_ORDER]
    counts = [group_counts[g] for g in PHENOMENA_ORDER]
    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(x_labels)), 3))
    ax.imshow([counts], cmap="Blues", aspect="auto")
    # Set x-ticks and labels
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=30, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(["# Phenomena"], rotation=0)
    # Annotate counts
    for i, count in enumerate(counts):
        ax.text(i, 0, str(count), va='center', ha='center', color='black', fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Phenomenon group counts image saved to {output_path}")

def plot_model_group_performance_table(
    csv_path: str = "experiments/output/v2/results.csv",
    output_dir: str = "analysis/output",
    output_filename: str = "model_group_performance_table.png"
):
    """
    Plots a table (heatmap) with models (MODEL_ORDER) as rows, phenomenon groups (PHENOMENA_ORDER) as columns,
    each cell showing the average accuracy for that model on all phenomena in the group.
    Always includes all groups in PHENOMENA_ORDER as columns, even if missing from the CSV.
    Uses PHENOMENA_ABR for x-axis labels. Saves PNG to output_dir.
    """
    df = pd.read_csv(csv_path)
    # Build a mapping: group -> list of phenomena
    group_to_phenomena = {group: [p for p, g in PHENOMENA_LIST_MAP.items() if g == group] for group in PHENOMENA_ORDER}
    # Prepare data for the table
    data = []
    for model in MODEL_ORDER:
        row = []
        for group in PHENOMENA_ORDER:
            phenomena = group_to_phenomena.get(group, [])
            # Only use phenomena that are present in the results.csv
            phenomena_in_csv = [p for p in phenomena if p in set(df["grammatical phenomenon"].unique())]
            if not phenomena_in_csv:
                row.append(float('nan'))
                continue
            accs = df[(df["model name"] == model) & (df["grammatical phenomenon"].isin(phenomena_in_csv))]["accuracy"]
            row.append(accs.mean() if not accs.empty else float('nan'))
        data.append(row)
    # Build DataFrame with all columns present (even if all NaN)
    columns = [PHENOMENA_ABR.get(g, g) for g in PHENOMENA_ORDER]
    df_table = pd.DataFrame(data, index=MODEL_ORDER, columns=columns)
    # Add 'Overall' column at the start (mean across groups, ignoring NaN)
    df_table.insert(0, 'Overall', df_table.mean(axis=1, skipna=True))
    # Plot heatmap
    plt.figure(figsize=(max(2 + len(columns)+1, 8), max(1 + len(MODEL_ORDER), 8)))
    ax = sns.heatmap(df_table, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, linewidths=0.5, linecolor='gray', annot_kws={"size": 10})
    ax.set_ylabel("Model")
    # Move x-ticks to top
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # Align xticks with columns (including 'Overall')
    ax.set_xticks([i + 0.5 for i in range(len(df_table.columns))])
    # ax.set_xticklabels(df_table.columns, rotation=30, ha='right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, output_filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Model-group performance table saved to {outpath}")

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
    # percent_non_english_best_model()
    # save_performance_table_image(PHENOMENA_LIST, output_filename="performance_table_overall.png")
    # save_performance_table_image([
    #         "anaphor_gender_agreement",
    #         "anaphor_number_agreement",
    #         "animate_subject_passive",
    #         "determiner_noun_agreement_with_adj_2",
    #         "existential_there_quantifiers_1"
    #     ], output_filename="performance_table_local.png")
    # save_performance_table_image([
    #     "adjunct_island",
    #     "distractor_agreement_relative_clause",
    #     "ellipsis_n_bar_1",
    #     "principle_A_c_command",
    #     "wh_questions_object_gap",
    #     "wh_questions_object_gap_long_distance",
    # ], output_filename="performance_table_structural.png")
    # save_performance_table_image([
    #     "wh_questions_subject_gap",
    #     "wh_questions_subject_gap_long_distance",
    # ], output_filename="performance_table_subject.png")
    # plot_model_group_performance_table()
    plot_phenomenon_group_counts()

if __name__ == "__main__":
    main()