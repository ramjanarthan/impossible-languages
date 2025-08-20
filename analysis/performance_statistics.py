from math import fabs
from analysis.dependency.dependency_parse import crossing_dependencies_count
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional, Dict

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

PHENOMENA_TO_DISPLAY_NAME = {
    "adjunct_island": "Adjunct Island",
    "anaphor_gender_agreement": "Anaphor Gender Agreement",
    "anaphor_number_agreement": "Anaphor Number Agreement",
    "animate_subject_passive": "Animate Subject Passive",
    "distractor_agreement_relative_clause": "Distractor Agreement Relative Clause",
    "ellipsis_n_bar_1": "Ellipsis N-bar 1",
    "irregular_past_participle_adjectives": "Irregular Past Participle Adjectives",
    "principle_A_c_command": "Principle A C-command",
    "wh_questions_object_gap": "WH Questions Object Gap",
    "wh_questions_object_gap_long_distance": "WH Questions Object Gap Long Distance",
    "wh_questions_subject_gap": "WH Questions Subject Gap",
    "wh_questions_subject_gap_long_distance": "WH Questions Subject Gap Long Distance",
    "left_branch_island_simple_question": "Left Branch Island Simple Question",
    "determiner_noun_agreement_with_adj_2": "Determiner Noun Agreement With Adj 2",
    "existential_there_quantifiers_1": "Existential There Quantifiers 1"
}

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

MODEL_TO_DISPLAY_NAME = {
    "english": "English",
    "reverse_full": "Full Reverse",
    "reverse_partial": "Partial Reverse",
    "shuffle_local3": "Local Shuffle3",
    "shuffle_local5": "Local Shuffle5",
    "shuffle_local10": "Local Shuffle10",
    "shuffle_even_odd": "Even-odd Shuffle",
    "shuffle_deterministic21": "Deterministic Shuffle21",
    "shuffle_nondeterministic": "Nondeterministic Shuffle",
}

def plot_ordering(
    labels: List[str],
    values: List[float],
    output_path: str = "analysis/output/ordering.png",
    title: str = "Model Performance Ranking",
    reverse: bool = True,
    col1_title: str = "Item",
    col2_title: str = "Value",
):
    """
    Generates a PNG image showing an ordered list of labels and their values.

    The plot width scales dynamically with the length of the labels to prevent
    text overflow. Text is left-aligned for readability.

    Args:
        labels (List[str]): A list of strings to be displayed (e.g., model names).
        values (List[float]): A list of corresponding numerical values (e.g., scores).
        output_path (str): The path where the generated image will be saved.
        title (str): The title for the plot.
    """
    if not labels or len(labels) != len(values):
        print("Error: Labels and values must be non-empty and have the same length.")
        return

    sorted_data = sorted(zip(labels, values), key=lambda item: item[1], reverse=reverse)
    sorted_labels, sorted_values = zip(*sorted_data)

    # Determine figure width dynamically based on the longest label
    if sorted_labels:
        max_len = max(len(label) for label in sorted_labels)
        # Base width of 4 inches + 0.12 inches per character for the label
        fig_width = 4 + max_len * 0.04
    else:
        fig_width = 5 # Default width if there are no labels

    # if column titles combined are bigger than fig_width, increase fig_width
    if len(col1_title) + len(col2_title) > 30:
        fig_width = fig_width + 2

    fig, ax = plt.subplots(figsize=(fig_width, 0.6 * (len(sorted_labels) + 1)))

    # Add column titles
    ax.text(-0.5, -1, f"  {col1_title}", va='center', ha='left', color='black', fontsize=14, fontweight='bold')
    ax.text(0.5, -1, f"{col2_title}  ", va='center', ha='right', color='black', fontsize=14, fontweight='bold')
    
    background = [[1] for _ in sorted_labels]
    ax.imshow(background, cmap="Blues", aspect="auto", vmin=0, vmax=1.5, extent=(-0.5, 0.5, len(sorted_labels)-0.5, -0.5))

    for i, (label, value) in enumerate(zip(sorted_labels, sorted_values)):
        rank = i + 1
        # Add padding to keep text from the edges
        text = f"  {rank}. {label}"
        
        # Display the rank and label, now LEFT-aligned
        ax.text(-0.5, i, text, va='center', ha='left', color='white', fontsize=14, fontweight='bold')
        
        # Display the formatted score, RIGHT-aligned
        ax.text(0.5, i, f"{value:.3f}  ", va='center', ha='right', color='white', fontsize=12, alpha=0.8)

    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Ordering plot saved to {output_path}")


def plot_model_ordering_from_csv(
    phenomena: List[str],
    skip_models: Optional[List[str]] = None,
    csv_path: str = "experiments/output/v2/results.csv",
    output_path: str = "analysis/output/model_ordering.png"
):
    """
    Calculates overall model performance on specific phenomena and plots their ordering.

    This function reads a CSV file, filters the data by the given phenomena, calculates
    the mean accuracy for each model (while omitting specified models), and then
    calls plot_ordering to generate the final visualization.

    Args:
        phenomena (List[str]): A list of grammatical phenomena to include.
        skip_models (Optional[List[str]]): A list of model names to exclude from the plot.
        csv_path (str): Path to the input results CSV file.
        output_path (str): The path to save the generated ordering plot.
    """
    if skip_models is None:
        skip_models = []
        
    df = pd.read_csv(csv_path)
    
    # Filter the DataFrame to include only the desired phenomena
    df_filtered = df[df["grammatical phenomenon"].isin(phenomena)]
    if df_filtered.empty:
        print(f"Warning: No data found for the specified phenomena.")
        return
        
    # Exclude any models specified in the skip_models list
    df_filtered = df_filtered[~df_filtered["model name"].isin(skip_models)]
    
    # Calculate the overall performance (mean accuracy) for each model
    performance = df_filtered.groupby("model name")["accuracy"].mean()
    
    # Extract the model names and their scores
    labels = performance.index.tolist()
    values = performance.values.tolist()
    
    # Call the plotting function to create and save the image
    plot_ordering(labels=labels, values=values, output_path=output_path, col1_title="Model", col2_title="Accuracy")


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
    
    # Pivot table: index=phenomenon, columns=model,    # Map to display names for table
    df["grammatical phenomenon"] = df["grammatical phenomenon"].apply(lambda x: PHENOMENA_TO_DISPLAY_NAME.get(x, x))
    df["model name"] = df["model name"].apply(lambda x: MODEL_TO_DISPLAY_NAME.get(x, x))
    phenomena_display = [PHENOMENA_TO_DISPLAY_NAME.get(p, p) for p in phenomena]
    all_models_display = [MODEL_TO_DISPLAY_NAME.get(m, m) for m in MODEL_ORDER]

    # Create pivot table
    table = df.pivot_table(index="grammatical phenomenon", columns="model name", values="accuracy")
    # Ensure all models/phenomena present (fill missing with NaN)
    table = table.reindex(index=phenomena_display, columns=all_models_display)

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

def plot_ranking_comparison(
    ranking1: List[str],
    ranking2: List[str],
    ranking1_label: str = "Ranking A",
    ranking2_label: str = "Ranking B",
    output_path: str = "analysis/output/ranking_comparison.png",
    title: str = "Comparison of Rankings"
):
    """
    Creates a slope graph to compare two different rankings of the same items.

    Args:
        ranking1 (List[str]): An ordered list of item names for the first ranking.
        ranking2 (List[str]): An ordered list of item names for the second ranking.
        ranking1_label (str): The name of the first ranking (for the column title).
        ranking2_label (str): The name of the second ranking.
        output_path (str): The path to save the generated image.
        title (str): The overall title for the plot.
    """
    # Create rank mappings for efficient lookup
    rank1_map = {item: i for i, item in enumerate(ranking1)}
    rank2_map = {item: i for i, item in enumerate(ranking2)}
    all_items = sorted(list(set(ranking1) | set(ranking2)))
    num_items = len(all_items)

    fig, ax = plt.subplots(figsize=(8, max(5, 0.45 * num_items)))

    # X-coordinates for the two rankings
    x1, x2 = 0, 1

    # Plot lines and labels for each item
    for item in all_items:
        # Items must be in both rankings to be compared
        if item not in rank1_map or item not in rank2_map:
            continue
            
        y1 = rank1_map[item]
        y2 = rank2_map[item]
        
        # Determine color based on change in rank
        if y1 > y2: color = 'green'    # Rank improved (e.g., 5th -> 2nd)
        elif y1 < y2: color = 'red'    # Rank worsened (e.g., 2nd -> 5th)
        else: color = 'darkgray'
        
        # Plot the connecting line
        ax.plot([x1, x2], [y1, y2], color=color, marker='o', markersize=8, alpha=0.7, linewidth=2)

        # Add text labels next to the points
        ax.text(x1 - 0.03, y1, f"{item} ({y1+1})", ha='right', va='center', fontsize=12)
        ax.text(x2 + 0.03, y2, f"({y2+1}) {item}", ha='left', va='center', fontsize=12)

    # Set column headers
    ax.text(x1, -0.5, ranking1_label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(x2, -0.5, ranking2_label, ha='center', va='bottom', fontsize=14, fontweight='bold')

    # --- Final plot styling ---
    ax.set_title(title, fontsize=16, fontweight='bold', pad=30)
    # Invert y-axis so Rank 1 is at the top
    ax.invert_yaxis()
    # Add padding to the y-axis
    ax.set_ylim(max(len(ranking1), len(ranking2)) - 0.5, -1)
    # Remove all axis ticks and spines for a cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Ranking comparison plot saved to {output_path}")

# --- NEW FUNCTION ---
def get_performance_ranking(
    phenomena: List[str],
    csv_path: str = "experiments/output/v2/results.csv",
    skip_models: Optional[List[str]] = None,
    reverse: bool = True
) -> List[str]:
    """
    Calculates model performance on specific phenomena and returns the ranked list of model names.

    Args:
        phenomena (List[str]): A list of grammatical phenomena to include.
        csv_path (str): Path to the input results CSV file.
        skip_models (Optional[List[str]]): A list of model names to exclude.
        reverse (bool): If True, sort descending (higher score is better).

    Returns:
        List[str]: An ordered list of model names from best to worst.
    """
    if skip_models is None:
        skip_models = []
        
    df = pd.read_csv(csv_path)
    df_filtered = df[df["grammatical phenomenon"].isin(phenomena)]
    if df_filtered.empty:
        print(f"Warning: No data found for the specified phenomena.")
        return []
        
    df_filtered = df_filtered[~df_filtered["model name"].isin(skip_models)]
    performance = df_filtered.groupby("model name")["accuracy"].mean()
    
    # Sort the performance Series and return the index (model names) as a list
    sorted_performance = performance.sort_values(ascending=not reverse)
    return sorted_performance.index.tolist()

# --- NEW FUNCTION ---
def plot_parallel_rankings(
    rankings: Dict[str, List[str]],
    output_path: str,
    should_enable_special_labelling_logic: bool = False,
    ranking_values: dict = None
):
    """
    Creates a parallel coordinates plot to compare multiple rankings of the same items.
    This visualization is excellent for showing consistency (parallel lines) or inconsistency
    (crossed lines) between different ranking criteria.
    Args:
        rankings (Dict[str, List[str]]): A dictionary where keys are ranking labels
                                       (e.g., "Accuracy") and values are ordered
                                       lists of item names (e.g., model names).
        output_path (str): The path to save the generated plot.
        title (str): The overall title for the plot.
    """
    ranking_labels = list(rankings.keys())
    num_rankings = len(ranking_labels)
    
    # 1. Get all unique items across all rankings
    all_items = sorted(list(set.union(*[set(r) for r in rankings.values()])))
    num_items = len(all_items)
    
    # 2. Create maps from item to rank for each ranking
    rank_maps = {}
    for label, ranking_list in rankings.items():
        rank_maps[label] = {item: i for i, item in enumerate(ranking_list)}
    
    fig, ax = plt.subplots(figsize=(4 * num_rankings, max(6, 0.5 * num_items)))

    # Map the custom labels to the model names used in the CSV file
    model_name_map = {
        "Base": "english", "Reverse": "reverse_full",
        "EvenOddShuffle": "shuffle_even_odd", "LocalShuffle(K=3)": "shuffle_local3",
        "LocalShuffle(K=5)": "shuffle_local5",
        "DeterministicShuffle": "shuffle_deterministic21"
    }
    
    # 3. Assign a unique color to each item for clear tracking
    colors = plt.cm.get_cmap('tab20', num_items)
    
    # 4. Plot lines for each item, but only connect items that exist in multiple rankings
    plotted_items = []
    for i, item in enumerate(all_items):
        # Find which rankings contain this item
        available_rankings = [label for label in ranking_labels if item in rank_maps[label]]
        
        # Only plot if item exists in at least 2 rankings
        if len(available_rankings) >= 2:
            x_positions = []
            y_positions = []
            
            # Get positions for this item across all rankings where it exists
            for j, label in enumerate(ranking_labels):
                if item in rank_maps[label]:
                    x_positions.append(j)
                    y_positions.append(rank_maps[label][item])
            
            # Plot the line connecting the ranks
            ax.plot(x_positions, y_positions, color=colors(i), marker='o',
                   markersize=8, alpha=0.9, linewidth=2.5, label=item)
            
            # Add text labels at the first and last positions where item appears
            if x_positions:
                first_x, first_y = x_positions[0], y_positions[0]
                last_x, last_y = x_positions[-1], y_positions[-1]
                
                # Append actual value in square brackets to the label if ranking_values is provided
                value_first = None
                value_last = None
                if ranking_values is not None:
                    # Try to get the value for this item at the first and last ranking
                    if ranking_labels[first_x] in ranking_values and item in ranking_values[ranking_labels[first_x]]:
                        value_first = ranking_values[ranking_labels[first_x]][item]
                    if ranking_labels[last_x] in ranking_values and item in ranking_values[ranking_labels[last_x]]:
                        value_last = ranking_values[ranking_labels[last_x]][item]
                label_first = f"({first_y+1}) {MODEL_TO_DISPLAY_NAME[model_name_map[item]] if should_enable_special_labelling_logic else item}"
                label_last = f"({last_y+1}) {item}"
                if value_first is not None:
                    label_first += f" [{value_first:.2f}]"
                if value_last is not None:
                    label_last += f" [{value_last:.2f}]"
                ax.text(first_x - 0.05, first_y, label_first, ha='right', va='center', fontsize=12)
                ax.text(last_x + 0.05, last_y, label_last, ha='left', va='center', fontsize=12)


            
            plotted_items.append(item)
    
    # 5. Add items that only appear in one ranking as single points
    for i, item in enumerate(all_items):
        if item not in plotted_items:
            # Find the single ranking this item appears in
            for j, label in enumerate(ranking_labels):
                if item in rank_maps[label]:
                    rank = rank_maps[label][item]
                    ax.plot(j, rank, color=colors(i), marker='o',
                           markersize=8, alpha=0.6)
                    ax.text(j + 0.05, rank, f"({rank+1}) {item}", 
                           ha='left', va='center', fontsize=12, alpha=0.6)
                    break
    
    # 6. Set column headers (the names of the rankings)
    max_rank = max(len(ranking) for ranking in rankings.values()) if rankings else 1
    for i, label in enumerate(ranking_labels):
        ax.text(i, -0.5, label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # 7. Final plot styling for a clean, professional look
    ax.invert_yaxis()  # Puts Rank 1 at the top
    ax.set_ylim(max_rank - 0.5, -1)
    ax.set_xlim(-0.5, num_rankings - 0.5)
    ax.set_xticks([])  # Hide x-axis ticks, we have headers
    ax.set_yticks([])  # Hide y-axis ticks
    ax.spines[:].set_visible(False)  # Remove the plot frame
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Parallel ranking plot saved to {output_path}")


    
def main():
    save_performance_table_image(PHENOMENA_LIST, output_filename="performance_table_overall.png")
    local_ordering_phenomena = [
        "anaphor_gender_agreement", "anaphor_number_agreement",
        "existential_there_quantifiers_1",
        "irregular_past_participle_adjectives",
        "wh_questions_subject_gap", "wh_questions_subject_gap_long_distance",
    ]
    save_performance_table_image(local_ordering_phenomena, output_filename="performance_table_local.png")

    structural_phenomena = [
        "adjunct_island",
        "animate_subject_passive",
        "determiner_noun_agreement_with_adj_2",
        "distractor_agreement_relative_clause",
        "ellipsis_n_bar_1",
        "principle_A_c_command",
        "wh_questions_object_gap",
        "wh_questions_object_gap_long_distance",
        "left_branch_island_simple_question",
    ]

    save_performance_table_image(structural_phenomena, output_filename="performance_table_structural.png")
    # save_performance_table_image([
    #     "wh_questions_subject_gap",
    #     "wh_questions_subject_gap_long_distance",
    # ], output_filename="performance_table_subject.png")
    # plot_model_group_performance_table()

    print("\n--- Generating plot for local ordering phenomena ---")
    
    plot_model_ordering_from_csv(
        phenomena=local_ordering_phenomena,
        # skip_models=["reverse_partial", "shuffle_nondeterministic"],
        output_path="analysis/output/model_ordering_local.png"
    )

    print("\n--- Generating plot from a custom list of models and scores ---")
    # Example of using the second function directly with custom data
    custom_labels = ["Base (english)", "Reverse (reverse_full)", "EvenOddShuffle (shuffle_even_odd)", "LocalShuffle(K=3) (shuffle_local3)", "LocalShuffle(K=5) (shuffle_local5)", "LocalShuffle(K=7)", "DeterministicShuffle (shuffle_deterministic21)"]
    custom_values = [2.92, 2.98, 3.76, 3.68, 3.88, 4.06, 4.60] # Note: The function will sort these automatically
    plot_ordering(
        labels=custom_labels,
        values=custom_values,
        output_path="analysis/output/m_local_ranking.png",
        title="m-local Ranking",
        col1_title="Impossible language",
        col2_title="m-local entropy",
        reverse=False
    )

    # --- LOCAL PHENOMENA: CONSISTENCY COMPARISON ---
    print("\n--- Generating comparison plot for local phenomena rankings ---")
    csv_file = "experiments/output/v2/results.csv"

    # Models and scores for the "m-local" metric
    mlocal_labels = ["Base", "Reverse", "EvenOddShuffle", "LocalShuffle(K=3)", "LocalShuffle(K=5)", "LocalShuffle(K=7)", "DeterministicShuffle"]
    mlocal_values = [2.92, 2.98, 3.76, 3.68, 3.88, 4.06, 4.60]
    
    # Map the custom labels to the model names used in the CSV file
    model_name_map = {
        "Base": "english", "Reverse": "reverse_full",
        "EvenOddShuffle": "shuffle_even_odd", "LocalShuffle(K=3)": "shuffle_local3",
        "LocalShuffle(K=5)": "shuffle_local5",
        "DeterministicShuffle": "shuffle_deterministic21"
    }

    skipped_local_models = ["reverse_partial", "shuffle_local10", "shuffle_nondeterministic"]

    # Reverse map for converting model names to display names
    display_name_map = {v: k for k, v in model_name_map.items()}
    
    # 1. Get Accuracy Ranking from CSV, convert to display names
    accuracy_ranking_local_models = get_performance_ranking(phenomena=local_ordering_phenomena, csv_path=csv_file)
    accuracy_ranking_local = [display_name_map.get(model, MODEL_TO_DISPLAY_NAME.get(model, model)) 
                             for model in accuracy_ranking_local_models 
                             if model in display_name_map or model in MODEL_TO_DISPLAY_NAME]
        
    # 2. Create m-local Ranking (lower score is better)
    sorted_mlocal = sorted(zip(mlocal_labels, mlocal_values), key=lambda item: item[1])
    mlocal_ranking = [model for model, _ in sorted_mlocal]

    ranking_values_local = {
        "Accuracy Ranking": accuracy_ranking_local_models,
        "m-local entropy Ranking": sorted_mlocal,
    }
    
    # 3. Plot the comparison
    plot_parallel_rankings(
        rankings={
            "Accuracy Ranking": accuracy_ranking_local,
            "m-local entropy Ranking": mlocal_ranking,
        },
        output_path="analysis/output/local_ranking_consistency.png",
        should_enable_special_labelling_logic=True,
        ranking_values=ranking_values_local
    )
    
    # --- STRUCTURAL PHENOMENA: INCONSISTENCY COMPARISON ---
    print("\n--- Generating comparison plot for structural phenomena rankings ---")
    
    # Models and their scores for other structural metrics
    structural_models = MODEL_ORDER
    dep_stats_filepath = "analysis/output/dep_stats.csv"
    df = pd.read_csv(dep_stats_filepath)
    grouped = df.groupby('perturbation')
    
    # Calculate aggregations
    agg_stats = {}
    for name, group in grouped:
        agg_stats[name] = {
            'avg_total_dep_distance': group['total_dependency_distance'].mean(),
            'avg_norm_dep_distance': group['normalized_dependency_distance'].mean(),
            'avg_crossing_deps': group['crossing_dependencies_count'].mean(),
            'proportion_projective': group['is_projective'].mean(),
            'num_sentences': len(group)
        }
    
    # Create a DataFrame from the aggregated stats
    stats_df = pd.DataFrame.from_dict(agg_stats, orient='index')
    dep_dist_values = stats_df['avg_norm_dep_distance'].tolist()
    projectivity_values = stats_df['proportion_projective'].tolist()
    crossing_dependencies_values = stats_df['avg_crossing_deps'].tolist()
    
    # Convert structural model names to display names
    structural_display_names = [MODEL_TO_DISPLAY_NAME.get(model, model) for model in structural_models]
    
    # 1. Get Accuracy Ranking from CSV, convert to display names
    accuracy_ranking_structural_models = get_performance_ranking(phenomena=structural_phenomena, csv_path=csv_file)
    accuracy_ranking_structural = [MODEL_TO_DISPLAY_NAME.get(model, model) for model in accuracy_ranking_structural_models]
    
    # 2. Create Dependency Distance Ranking (lower score is better)
    sorted_dep_dist = sorted(zip(structural_display_names, dep_dist_values), key=lambda item: item[1])
    dep_dist_ranking = [model for model, value in sorted_dep_dist]
    
    # 3. Create Projectivity Ranking (higher score is better)
    sorted_projectivity = sorted(zip(structural_display_names, projectivity_values), key=lambda item: item[1], reverse=True)
    projectivity_ranking = [model for model, _ in sorted_projectivity]
    
    # 4. Crossing dependency count (lower score is better)
    sorted_crossing = sorted(zip(structural_display_names, crossing_dependencies_values), key=lambda item: item[1])
    crossing_dependencies_ranking = [model for model, _ in sorted_crossing]

    sorted_mlocal = sorted(zip(mlocal_labels, mlocal_values), key=lambda item: item[1])
    mlocal_ranking = [model for model, _ in sorted_mlocal]

    # Value dictionaries for each ranking
    # accuracy_structural_values = {MODEL_TO_DISPLAY_NAME.get(model, model): value
    #                             for model, value in zip(accuracy_ranking_structural_models, [None]*len(accuracy_ranking_structural_models))}  # Replace None with actual values if available
    # dep_dist_values_dict = dict(sorted_dep_dist)
    # projectivity_values_dict = dict(sorted_projectivity)
    ranking_values_structural = {
        "Model Accuracy (Structural tasks)": accuracy_ranking_structural_models,
        "Projectivity":  sorted_projectivity,
        "Norm. Dep. Distance": sorted_dep_dist,
    }    
    # 5. Plot the comparison
    plot_parallel_rankings(
        rankings={
            "Accuracy Ranking": accuracy_ranking_structural,
            "Projectivity Ranking": projectivity_ranking,
            "Dependency Distance Ranking": dep_dist_ranking,
        },
        output_path="analysis/output/structural_ranking_inconsistency.png",
        ranking_values=ranking_values_structural,
    )

    # Structural ordering
    
    plot_model_ordering_from_csv(
        phenomena=structural_phenomena,
        output_path="analysis/output/model_ordering_structural.png"
    )    
    
    plot_ordering(
        labels=structural_models,
        values=dep_dist_values,
        output_path="analysis/output/mean_normalised_dep_distance_ranking.png",
        title="Mean Normalized Dependency Distance Ranking",
        col1_title="Impossible language model",
        col2_title="Mean Norm. Dep. Distance",
        reverse=False
    )

    plot_ordering(
        labels=structural_models,
        values=projectivity_values,
        output_path="analysis/output/projectivity_ranking.png",
        title="Projectivity Ranking",
        col1_title="Impossible language model",
        col2_title="Proportion Projective",
    )

    plot_ordering(
        labels=structural_models,
        values=crossing_dependencies_values,
        output_path="analysis/output/crossing_dependencies_ranking.png",
        title="Crossing Dependencies Ranking",
        col1_title="Impossible language model",
        col2_title="Crossing Dependencies",
        reverse=False
    )

if __name__ == "__main__":
    main()