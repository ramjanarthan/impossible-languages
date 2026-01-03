
import re
import os
import argparse
import math

def parse_log_file(log_file_path):
    """Parses the filtering log file and returns a list of dictionaries containing dataset stats."""
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return []

    datasets_stats = []
    
    with open(log_file_path, 'r') as f:
        content = f.read()

    # Split by the separator line
    blocks = content.split('----------------------------------------')

    for block in blocks:
        # Extract dataset name
        # Looking for lines like: Filtering data_generation/outputs/impossible_blimp/v3/adjunct_island_20260101_170829.jsonl
        dataset_match = re.search(r'Filtering\s+(.+?)\.jsonl', block)
        
        # Extract filtered count and total count
        # Looking for lines like: Filtered 0 sentences out of 1000
        stats_match = re.search(r'Filtered\s+(\d+)\s+sentences\s+out\s+of\s+(\d+)', block)

        if dataset_match and stats_match:
            full_path = dataset_match.group(1)
            dataset_name = os.path.basename(full_path)
            filtered_count = int(stats_match.group(1))
            total_count = int(stats_match.group(2))
            
            datasets_stats.append({
                'dataset': dataset_name,
                'filtered_sentences': filtered_count,
                'total_sentences': total_count,
                'percentage_filtered': (filtered_count / total_count) * 100 if total_count > 0 else 0
            })
    
    return datasets_stats

def calculate_percentile(data, percentile):
    """Calculates percentile from a list of numbers."""
    if not data:
        return 0
    size = len(data)
    sorted_data = sorted(data)
    k = (size - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1

def print_table(data, headers, log_func):
    """Prints a formatted table using standard libraries."""
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in data:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Print Header
    row_format = " | ".join([f"{{:<{w}}}" for w in widths])
    log_func("-" * (sum(widths) + 3 * (len(headers) - 1)))
    log_func(row_format.format(*headers))
    log_func("-" * (sum(widths) + 3 * (len(headers) - 1)))
    
    # Print Rows
    for row in data:
        log_func(row_format.format(*[str(val) for val in row]))
    log_func("-" * (sum(widths) + 3 * (len(headers) - 1)))

def print_statistics(datasets_stats, output_file=None):
    """Calculates and prints the statistics, optionally writing to a file."""
    
    # Helper to print to both stdout and file if it exists
    file_handle = None
    if output_file:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            file_handle = open(output_file, 'w')
        except Exception as e:
            print(f"Warning: Could not open output file {output_file}: {e}")

    def log(msg=""):
        print(msg)
        if file_handle:
            file_handle.write(msg + "\n")

    if not datasets_stats:
        log("No dataset statistics found.")
        if file_handle: file_handle.close()
        return

    # 1. Table of all datasets
    log("\n### Dataset Filtering Statistics ###\n")
    table_data = [[d['dataset'], d['filtered_sentences'], d['total_sentences'], f"{d['percentage_filtered']:.2f}%"] for d in datasets_stats]
    headers = ["Dataset Name", "Filtered Sentences", "Total Sentences", "% Filtered"]
    print_table(table_data, headers, log)

    # 2. Aggregate Statistics
    filtered_counts = [d['filtered_sentences'] for d in datasets_stats]
    datasets_with_filtering = [d for d in datasets_stats if d['filtered_sentences'] > 0]
    
    percent_with_filtering = (len(datasets_with_filtering) / len(datasets_stats)) * 100
    
    log("\n### Aggregate Statistics ###\n")
    log(f"Total Datasets: {len(datasets_stats)}")
    log(f"Datasets with ANY filtering: {len(datasets_with_filtering)} ({percent_with_filtering:.2f}%)")
    
    if filtered_counts:
        p25 = calculate_percentile(filtered_counts, 25)
        p50 = calculate_percentile(filtered_counts, 50)
        p75 = calculate_percentile(filtered_counts, 75)
        p90 = calculate_percentile(filtered_counts, 90)
        p99 = calculate_percentile(filtered_counts, 99)
        
        log("\n### Filtered Sentences Percentiles ###")
        log(f"P25: {p25:.2f}")
        log(f"P50 (Median): {p50:.2f}")
        log(f"P75: {p75:.2f}")
        log(f"P90: {p90:.2f}")
        log(f"P99: {p99:.2f}")
    
    # Extra: Identify heavy filtering
    log("\n### Top 5 Most Filtered Datasets ###")
    sorted_by_filtered = sorted(datasets_stats, key=lambda x: x['filtered_sentences'], reverse=True)[:5]
    top_table = [[d['dataset'], d['filtered_sentences'], f"{d['percentage_filtered']:.2f}%"] for d in sorted_by_filtered]
    print_table(top_table, headers=["Dataset", "Filtered Count", "% Filtered"], log_func=log)

    if file_handle:
        file_handle.close()
        print(f"\nAnalysis written to: {output_file}")

if __name__ == "__main__":
    LOG_FILE = "data_generation/outputs/impossible_blimp/v3_batch_filter_output.txt"
    OUTPUT_FILE = "data_generation/outputs/impossible_blimp/v3_batch_filter_output_analysis.txt"
    
    parser = argparse.ArgumentParser(description="Analyze filtering logs.")
    parser.add_argument("logfile", nargs="?", default=LOG_FILE, help="Path to the log file")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Path to the output analysis file")
    args = parser.parse_args()
    
    stats = parse_log_file(args.logfile)
    print_statistics(stats, output_file=args.output)
