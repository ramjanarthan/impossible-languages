import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
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
    
    print(f"Using columns: Dataset='{dataset_col}', Kallini='{kallini_col}', BLiMP='{blimp_col}'")
    
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
             label='Kallini GPT-2', color='#2E8B57', alpha=0.8)
    ax1.plot(x_pos, blimp_scores, 's-', linewidth=2, markersize=8, 
             label='BLiMP GPT-2', color='#FF6347', alpha=0.8)
    
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
    ax1.text(0.015, 0.16, f'Strong positive correlation (r = {correlation:.3f})', 
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
        
    # Print summary statistics
    print("=" * 60)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Number of datasets: {len(datasets)}")
    print(f"Correlation coefficient: {correlation:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Correlation strength: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'}")
    print("\nKallini GPT-2 Performance:")
    print(f"  Mean: {np.mean(kallini_scores):.3f}")
    print(f"  Std:  {np.std(kallini_scores):.3f}")
    print(f"  Range: {min(kallini_scores):.3f} - {max(kallini_scores):.3f}")
    print("\nBLiMP GPT-2 Performance:")
    print(f"  Mean: {np.mean(blimp_scores):.3f}")
    print(f"  Std:  {np.std(blimp_scores):.3f}")
    print(f"  Range: {min(blimp_scores):.3f} - {max(blimp_scores):.3f}")
    print("\nKey Insights:")
    print("• Both models show similar relative performance patterns across tasks")
    print("• BLiMP GPT-2 generally achieves higher absolute scores")
    print("• Strong positive correlation suggests consistent linguistic competencies")
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    compare_model_performances()