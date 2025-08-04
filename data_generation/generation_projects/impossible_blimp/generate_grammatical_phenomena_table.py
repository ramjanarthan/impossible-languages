import argparse
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

def generate_grammatical_phenomena_classification_table():
    
    # read from grammatical_phenomena.csv table
    table_data = pd.read_csv('data_generation/generation_projects/impossible_blimp/grammatical_phenomena_table.csv')
    
    if table_data.empty:
        print("No data to display")
        return

    num_rows = len(table_data)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 4))  # Adjust size based on content
    ax.axis('tight')
    ax.axis('off')

    the_table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.3, 0.2, 0.3, 0.3]
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 2.5)

    # Style the table
    for (i, _), cell in the_table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e0e0e0')  # light gray
            cell.set_fontsize(12)
        else:
            cell.set_edgecolor('lightgrey')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)

    plt.tight_layout(pad=2.0)

    output_path = 'data_generation/generation_projects/impossible_blimp/grammatical_phenomena_table.png'

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Table saved to {output_path}")

def main():
    generate_grammatical_phenomena_classification_table()

if __name__ == "__main__":
    main()
 