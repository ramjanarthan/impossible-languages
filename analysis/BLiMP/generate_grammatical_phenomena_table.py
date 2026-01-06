import argparse
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

EXCEPTIONAL_STRONG_PHENOMENA = set(["anaphor_gender_agreement", "anaphor_number_agreement"])
EXCEPTIONAL_WEAK_PHENOMENA = set() #set(["left_branch_island_echo_question"])

def classify_phenomena(threshold: int = 80, 
                       raw_scores_path: str = 'analysis/BLiMP/blimp_raw_scores.csv',
                       output_path: str = 'analysis/BLiMP/grammatical_phenomena_table.csv'):
    """
    Classify phenomena as 'Strong' or 'Weak' based on 5-gram scores.
    
    Parameters:
    -----------
    threshold : int
        The threshold for classification. Scores >= threshold are 'Strong', otherwise 'Weak'.
    raw_scores_path : str
        Path to the CSV file containing raw BLiMP scores with a '5-gram' column.
    output_path : str
        Path to write the classification results CSV.
    """
    # Read raw scores
    df = pd.read_csv(raw_scores_path)
    
    # Convert 5-gram column to numeric, handling any non-numeric values
    df['5-gram'] = pd.to_numeric(df['5-gram'], errors='coerce')
    
    # Classify based on threshold, with exceptions always classified as Strong
    def classify_row(row):
        if row['UID'] in EXCEPTIONAL_STRONG_PHENOMENA:
            return 'Strong'
        elif row['UID'] in EXCEPTIONAL_WEAK_PHENOMENA:
            return 'Weak'
        return 'Strong' if pd.notna(row['5-gram']) and row['5-gram'] >= threshold else 'Weak'
    
    df['Classification'] = df.apply(classify_row, axis=1)
    
    # Create output dataframe with relevant columns
    output_df = df[['Phenomenon', 'UID', '5-gram', 'Classification']].copy()
    output_df.columns = ['Phenomenon', 'Dataset Name', '5-gram Score', 'Cue Reliability']
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Classification saved to {output_path}")
    print(f"  - Strong (>= {threshold}): {(output_df['Cue Reliability'] == 'Strong').sum()}")
    print(f"  - Weak (< {threshold}): {(output_df['Cue Reliability'] == 'Weak').sum()}")
    
    return output_df


def generate_grammatical_phenomena_classification_table(
        input_path: str = 'analysis/BLiMP/grammatical_phenomena_table.csv',
        output_path: str = 'analysis/output/grammatical_phenomena_table.png'):
    """
    Generate a table image from the grammatical phenomena classification CSV.
    
    Parameters:
    -----------
    input_path : str
        Path to the CSV file containing the classification data.
    output_path : str
        Path to save the output PNG image.
    """
    # Read from grammatical_phenomena_table.csv
    table_data = pd.read_csv(input_path)
    
    if table_data.empty:
        print("No data to display")
        return

    # Calculate summary stats before adding summary row
    if 'Cue Reliability' in table_data.columns:
        strong_count = (table_data['Cue Reliability'] == 'Strong').sum()
        weak_count = (table_data['Cue Reliability'] == 'Weak').sum()
        
        # Add summary row at the end
        summary_row = pd.DataFrame({
            table_data.columns[0]: ['TOTAL'],
            table_data.columns[1]: [f'Strong: {strong_count}, Weak: {weak_count}'],
            table_data.columns[2]: [''],
            table_data.columns[3]: ['']
        })
        table_data = pd.concat([table_data, summary_row], ignore_index=True)

    num_rows = len(table_data)

    # Create figure and axes - adjust height based on number of rows
    fig_height = max(4, num_rows * 0.4)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')

    # Calculate column widths based on number of columns
    # Phenomenon: 0.25, Dataset Name: 0.5 (wider for long names), others: 0.15
    num_cols = len(table_data.columns)
    col_widths = [0.25, 0.4, 0.15, 0.15]  # Phenomenon, Dataset Name, 5-gram Score, Cue Reliability

    the_table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='left',
        loc='center',
        colWidths=col_widths[:num_cols]
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 2.0)

    # Style the table
    summary_row_idx = num_rows  # Last row (1-indexed because of header)
    for (i, j), cell in the_table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')  # light gray
            cell.set_fontsize(12)
        elif i == summary_row_idx:  # Summary row
            cell.set_facecolor('#f5f5f5')  # very light gray
            cell.set_text_props(weight='bold')
        else:
            cell.set_edgecolor('lightgrey')
            # Color code the Cue Reliability column
            if j == table_data.columns.get_loc('Cue Reliability') if 'Cue Reliability' in table_data.columns else -1:
                value = table_data.iloc[i - 1, j]
                if value == 'Strong':
                    cell.set_facecolor('#c8e6c9')  # light green
                elif value == 'Weak':
                    cell.set_facecolor('#ffcdd2')  # light red
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

    plt.tight_layout(pad=2.0)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Table saved to {output_path}")


def main():
    # First, classify phenomena based on human scores
    classify_phenomena()
    
    # Then, generate the table image
    generate_grammatical_phenomena_classification_table()


if __name__ == "__main__":
    main()