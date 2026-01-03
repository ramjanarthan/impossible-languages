import pandas as pd
import os

def compare_results():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2_path = os.path.join(base_dir, "experiments", "output", "v2", "results.csv")
    v3_path = os.path.join(base_dir, "experiments", "output", "v3", "results.csv")

    # Check if files exist
    if not os.path.exists(v2_path):
        print(f"Error: V2 results not found at {v2_path}")
        return
    if not os.path.exists(v3_path):
        print(f"Error: V3 results not found at {v3_path}")
        return

    # Load data
    print(f"Loading V2 results from: {v2_path}")
    df_v2 = pd.read_csv(v2_path)
    print(f"Loading V3 results from: {v3_path}")
    df_v3 = pd.read_csv(v3_path)

    # Clean column names (strip whitespace)
    df_v2.columns = df_v2.columns.str.strip()
    df_v3.columns = df_v3.columns.str.strip()

    # Define merge columns
    merge_cols = ['grammatical phenomenon', 'model name', 'checkpoint']
    
    # Select strictly relevant columns for comparison + merge keys
    # Columns to compare: accuracy, perplexity good, perplexity bad
    value_cols = ['accuracy', 'perplexity good', 'perplexity bad']
    
    cols_to_keep = merge_cols + value_cols

    # Filter dataframes to keep only necessary columns to avoid conflicts on other columns
    df_v2_subset = df_v2[cols_to_keep].copy()
    df_v3_subset = df_v3[cols_to_keep].copy()

    # Merge
    print("Merging datasets...")
    merged = pd.merge(
        df_v2_subset, 
        df_v3_subset, 
        on=merge_cols, 
        suffixes=('_v2', '_v3'), 
        how='inner'
    )

    # Sort by grammatical phenomenon, model name, checkpoint
    merged = merged.sort_values(by=['grammatical phenomenon', 'model name', 'checkpoint'])

    # Calculate differences
    merged['acc_diff'] = merged['accuracy_v3'] - merged['accuracy_v2']
    merged['perp_good_diff'] = merged['perplexity good_v3'] - merged['perplexity good_v2']
    merged['perp_bad_diff'] = merged['perplexity bad_v3'] - merged['perplexity bad_v2']

    # Reorder columns for better readability
    final_cols = [
        'grammatical phenomenon', 'model name', 'checkpoint',
        'accuracy_v2', 'accuracy_v3', 'acc_diff',
        'perplexity good_v2', 'perplexity good_v3', 'perp_good_diff',
        'perplexity bad_v2', 'perplexity bad_v3', 'perp_bad_diff'
    ]
    
    final_df = merged[final_cols]

    # Save to CSV
    output_path = os.path.join(base_dir, "experiments", "output", "v3_vs_v2_comparison.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Comparison saved to: {output_path}")

    # Display preview
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 1000)
    # print("\nComparison Preview (First 20 rows):")
    # print(final_df.head(20))

    # Print summary of differences
    print("\n--- Summary by Grammatical Phenomenon ---")
    print(f"Total common rows: {len(final_df)}")

    # Check for differences
    diff_mask = (final_df['acc_diff'].abs() > 1e-9) | \
                (final_df['perp_good_diff'].abs() > 1e-9) | \
                (final_df['perp_bad_diff'].abs() > 1e-9)
    
    diff_rows = final_df[diff_mask]
    
    if diff_rows.empty:
        print("\nSUCCESS: All common rows are perfectly aligned (no differences in accuracy or perplexity).")
    else:
        print(f"\nFound {len(diff_rows)} rows with differences.")
        
        # Group by grammatical phenomenon to show where differences are
        grouped = diff_rows.groupby('grammatical phenomenon').agg({
            'acc_diff': ['count', 'mean', 'max'],
            'perp_good_diff': ['mean', 'max'],
            'perp_bad_diff': ['mean', 'max']
        })
        
        print("\n--- Differences by Phenomenon ---")
        print(grouped)
        
        print("\n--- Detailed Differences (First 10) ---")
        print(diff_rows.head(10)[['grammatical phenomenon', 'model name', 'checkpoint', 'acc_diff', 'perp_good_diff', 'perp_bad_diff']])

if __name__ == "__main__":
    compare_results()
