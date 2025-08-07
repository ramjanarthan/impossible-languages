#!/usr/bin/env python3
"""
Script to sample sentence pairs from a dataset and present them in a table, saved as a PNG image.

Usage:
    python -m analysis.sample_pair_table <path/to/dataset.jsonl> <output.png> \
        --samples <N> \
        --seed <SEED> \
        --perturbations <p1> <p2> ...
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to the Python path to allow for absolute imports
# This is necessary to run the script as a module from the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from data_generation.generation_projects.impossible_blimp.sample_dataset import (
    get_other_versions,
    load_dataset,
    sample_examples,
    VALID_PERTURBATION_KEYS
)

def generate_table_image(
    samples,
    perturbations_to_keep,
    output_path,
    title='Sampled Sentence Pairs'
):
    """Generates and saves a PNG table of sentence pairs."""
    if not samples:
        print("No samples to display.")
        return

    # Filter and structure the data for the table
    table_data = []
    
    # Create a set for faster lookup
    perturbations_set = set(perturbations_to_keep)

    # Maintain a list of processed sample indices to avoid duplicates from different perturbations
    processed_indices = set()

    for sample in samples:
        if sample['index'] in processed_indices:
            continue

        # Process English first if requested
        if "english" in perturbations_set:
            pair_text = f"Grammatical: {sample['english']['sentence_good']}\nUngrammatical:  {sample['english']['sentence_bad']}"
            table_data.append(["English", pair_text])

        # Process other versions
        for version, sentences in sample['other_versions'].items():
            if version in perturbations_set:
                display_version = version.replace('_', ' ').title()
                pair_text = f"Grammatical: {sentences['sentence_good']}\nUngrammatical:  {sentences['sentence_bad']}"
                table_data.append([display_version, pair_text])
        
        processed_indices.add(sample['index'])


    if not table_data:
        print("No data to display after filtering for perturbations.")
        return

    num_rows = len(table_data)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, num_rows * 1.5))  # Adjust size based on content
    ax.axis('tight')
    ax.axis('off')

    columns = ('Language', 'Example Pair')
    the_table = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc='left',
        loc='center',
        colWidths=[0.2, 0.5]
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

    # ax.set_title(title, weight='bold', size=16, y=1.05)
    plt.tight_layout(pad=2.0)

    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Table saved to {output_path}")

    # --- Crop the image to remove excess whitespace ---
    try:
        from PIL import Image
        im = Image.open(output_path)
        width, height = im.size
        left = int(width * 0.12)
        upper = int(height * 0.28)
        right = int(width * 0.88)
        lower = int(height * 0.72)
        im_cropped = im.crop((left, upper, right, lower))
        im_cropped.save(output_path)
        print(f"Cropped image saved to {output_path}")
    except Exception as e:
        print(f"Warning: Could not crop image: {e}")

def main():
    parser = argparse.ArgumentParser(description='Sample sentence pairs and save as a PNG table.')
    parser.add_argument('dataset_path', type=str, help='Path to the English version of the dataset (e.g., anaphor_agreement.jsonl)')
    parser.add_argument('output_path', type=str, help='Path to save the output PNG image.')
    parser.add_argument('--samples', type=int, default=3, help='Number of unique sentence pairs to sample.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument(
        '--perturbations',
        nargs='+',
        default=VALID_PERTURBATION_KEYS,
        choices=VALID_PERTURBATION_KEYS,
        help='A list of perturbations to include in the table.'
    )

    args = parser.parse_args()

    # Find all dataset versions
    dataset_files = get_other_versions(args.dataset_path)
    if not dataset_files:
        print(f"Error: No dataset files found for base: {args.dataset_path}")
        return

    # Load all datasets
    datasets = {name: load_dataset(path) for name, path in dataset_files.items()}

    # Sample examples
    samples = sample_examples(datasets, num_samples=args.samples, seed=args.seed)

    # Generate and save the table
    title = f'Sampled Pairs from {Path(args.dataset_path).stem}'
    generate_table_image(samples, args.perturbations, args.output_path, title)

if __name__ == "__main__":
    main()
 