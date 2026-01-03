import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from analysis.performance_statistics import MODEL_ORDER, MODEL_TO_DISPLAY_NAME, PHENOMENA_LIST_MAP, PHENOMENA_ORDER, PHENOMENA_ABR
import os

def compare_model_performances():
    """
    Display a comparison between Kallini GPT-2 and BLiMP GPT-2 model performances
    across different linguistic datasets, highlighting their similar trends.
    Reads data from CSV file and saves comparison plot.
    """
    
    # Read data from CSV file
    try:
        df = pd.read_csv("analysis/blimp_comparison.csv")
    except FileNotFoundError:
        print("Error: Could not find 'analysis/blimp_comparison.csv'")
        print("Please ensure the CSV file exists")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Try different possible column name variations
    dataset_col = 'Dataset'
    kallini_col = 'Kallini'
    blimp_col = 'BLiMP'
    
    # Check if all columns were found
    if not all([dataset_col, kallini_col, blimp_col]):
        print("Error: Could not identify the required columns.")
        print(f"Expected columns containing: {dataset_col}, {kallini_col}, {blimp_col}")
        print(f"Found columns: {list(df.columns)}")
        return
    
    
    # Extract data from DataFrame
    datasets = df[dataset_col].tolist()
    kallini_scores = df[kallini_col].tolist()
    blimp_scores = df[blimp_col].tolist()
    
    # Calculate correlation
    correlation, p_value = pearsonr(kallini_scores, blimp_scores)
    
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(14, 12))
    
    # Plot 1: Line plot showing trends
    x_pos = np.arange(len(datasets))
    
    ax1.plot(x_pos, kallini_scores, 'o-', linewidth=2, markersize=8, 
             label='Kallini GPT-2 model', color='#2E8B57', alpha=0.8)
    ax1.plot(x_pos, blimp_scores, 's-', linewidth=2, markersize=8, 
             label='BLiMP GPT-2 model', color='#FF6347', alpha=0.8)
    
    ax1.set_xlabel('Datasets', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    # ax1.set_title('Model Performance Comparison Across Linguistic Tasks\n' + 
    #               f'Correlation: r = {correlation:.3f} (p = {p_value:.4f})', 
    #               fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=16)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=16)
    
    # Add correlation annotation
    ax1.text(0.015, 0.16, f'Strong positive correlation (r = {correlation:.3f}) (p = {p_value:.4f})', 
             transform=ax1.transAxes, fontsize=16, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
             verticalalignment='top')
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = "analysis/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot
    output_path = os.path.join(output_dir, "blimp_comparison.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
        
    # # Print summary statistics
    # print("=" * 60)
    # print("MODEL PERFORMANCE COMPARISON SUMMARY")
    # print("=" * 60)
    # print(f"Number of datasets: {len(datasets)}")
    # print(f"Correlation coefficient: {correlation:.4f}")
    # print(f"P-value: {p_value:.6f}")
    # print(f"Correlation strength: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'}")
    # print("\nKallini GPT-2 Performance:")
    # print(f"  Mean: {np.mean(kallini_scores):.3f}")
    # print(f"  Std:  {np.std(kallini_scores):.3f}")
    # print(f"  Range: {min(kallini_scores):.3f} - {max(kallini_scores):.3f}")
    # print("\nBLiMP GPT-2 Performance:")
    # print(f"  Mean: {np.mean(blimp_scores):.3f}")
    # print(f"  Std:  {np.std(blimp_scores):.3f}")
    # print(f"  Range: {min(blimp_scores):.3f} - {max(blimp_scores):.3f}")
    # print("\nKey Insights:")
    # print("• Both models show similar relative performance patterns across tasks")
    # print("• BLiMP GPT-2 generally achieves higher absolute scores")
    # print("• Strong positive correlation suggests consistent linguistic competencies")
    # print("=" * 60)

def snake_to_sentence_case(snake_str):
    return ' '.join(word.capitalize() for word in snake_str.split('_'))

def plot_blimp_heatmap(model_results_path='experiments/output/v2/results.csv', 
                       blimp_aggr_scores_path='analysis/blimp_aggr_scores.csv',
                       output_filename='blimp_comparison_heatmap.png'):
    """
    Plot a heatmap comparing model performances with human scores on BLiMP grammatical phenomena tasks.
    
    Parameters:
    -----------
    model_results_path : str
        Path to the CSV file containing model results
    human_scores_path : str  
        Path to the CSV file containing human scores
    output_dir : str
        Directory to save the output plot
    output_filename : str
        Filename for the saved plot
    """

    # Read model results
    df_models = pd.read_csv(model_results_path)
    
    # Read BLiMP aggregate scores (contains Human row)
    df_blimp = pd.read_csv(blimp_aggr_scores_path)
    
    # Build a mapping: group -> list of phenomena
    group_to_phenomena = {group: [p for p, g in PHENOMENA_LIST_MAP.items() if g == group] 
                         for group in PHENOMENA_ORDER}
    
    # Prepare model data
    model_data = []
    for model in MODEL_ORDER:
        row = []
        for group in PHENOMENA_ORDER:
            phenomena = group_to_phenomena.get(group, [])
            # Only use phenomena that are present in the results.csv
            phenomena_in_csv = [p for p in phenomena if p in set(df_models["grammatical phenomenon"].unique())]
            if not phenomena_in_csv:
                row.append(float('nan'))
                continue
            accs = df_models[(df_models["model name"] == model) & 
                           (df_models["grammatical phenomenon"].isin(phenomena_in_csv))]["accuracy"]
            row.append(accs.mean() if not accs.empty else float('nan'))
        model_data.append(row)
    
    # Create model display names for rows
    model_display_names = [MODEL_TO_DISPLAY_NAME.get(model, model) for model in MODEL_ORDER]
    
    # Build model DataFrame
    df_model_table = pd.DataFrame(model_data, index=model_display_names, columns=PHENOMENA_ORDER)
    
    # Prepare human data row using PHENOMENA_ABR mapping
    # The blimp_aggr_scores.csv has column names like 'ISLAND', 'ANA. AGR', etc.
    human_row_data = df_blimp[df_blimp['Model'] == 'Human'].iloc[0]
    human_row = []
    for group in PHENOMENA_ORDER:
        # Map the phenomenon name to its abbreviation used in the CSV columns
        col_name = PHENOMENA_ABR.get(group)
        if col_name and col_name in human_row_data.index:
            # Convert from percentage (e.g., 84.9) to proportion (0.849)
            human_row.append(human_row_data[col_name] / 100.0)
        else:
            human_row.append(float('nan'))
    
    # Add human scores as the first row
    df_human_row = pd.DataFrame([human_row], index=['Human'], columns=PHENOMENA_ORDER)
    
    # Combine model and human data
    df_combined = pd.concat([df_human_row, df_model_table])
    
    # Add 'Overall' column at the start (mean across groups, ignoring NaN)
    df_combined.insert(0, 'Overall', df_combined.mean(axis=1, skipna=True))

    # Skip the 'control_raise' and 'npi' columns
    columns_to_drop = [col for col in ['control_raise', 'npi'] if col in df_combined.columns]
    if columns_to_drop:
        df_combined = df_combined.drop(columns=columns_to_drop)
    
    # Drop specific model rows
    # models_to_drop = [ 'Local Shuffle5', 'Local Shuffle10'] #'Deterministic Shuffle21', 'Nondeterministic Shuffle']
    # rows_to_drop = [model for model in models_to_drop if model in df_combined.index]
    # if rows_to_drop:
    #     df_combined = df_combined.drop(index=rows_to_drop)
    
    # Plot heatmap
    plt.figure(figsize=(max(2 + len(PHENOMENA_ORDER) + 1, 12), max(1 + len(MODEL_ORDER) + 1, 10)))
    
    # Convert snake_case column names to Sentence Case for display
    display_columns = [col if col == 'Overall' else snake_to_sentence_case(col) for col in df_combined.columns]
    
    # Create heatmap
    ax = sns.heatmap(df_combined, annot=True, fmt=".3f", cmap="YlGnBu", cbar=True, 
                     linewidths=0.5, linecolor='gray', annot_kws={"size": 9})
    
    ax.set_xlabel("Grammatical Phenomena", fontsize=16)
    ax.set_ylabel("Model/Human", fontsize=16)
    
    ax.xaxis.set_ticks_position('top') 
    ax.xaxis.set_label_position('top')
    
    # Align xticks with columns and set display names
    ax.set_xticks([i + 0.5 for i in range(len(df_combined.columns))])
    ax.set_xticklabels(display_columns, rotation=45, ha='left', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = "analysis/output"
    os.makedirs(output_dir, exist_ok=True)
    
    outpath = os.path.join(output_dir, output_filename)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model-human comparison heatmap saved to {outpath}")
    
    return df_combined
    
# Example usage
if __name__ == "__main__":
    compare_model_performances()
    plot_blimp_heatmap()