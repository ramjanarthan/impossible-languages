"""
Count the number of entries in each English (filtered) dataset from master_dataset_list.txt,
and add a '% Discarded' column to grammatical_phenomena_table.csv.

Expected total per dataset is 1000. If a dataset has N entries, % Discarded = (1000 - N) / 1000 * 100.
"""

import os
import csv
import re

# ---- Paths ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MASTER_LIST = os.path.join(PROJECT_ROOT, "data_generation", "generation_projects", "impossible_blimp", "master_dataset_list.txt")
CSV_PATH = os.path.join(PROJECT_ROOT, "analysis", "BLiMP", "grammatical_phenomena_table.csv")

EXPECTED_TOTAL = 1000

# ---- Step 1: Read master dataset list and count entries for English datasets ----
# English datasets are the ones ending in %filtered.jsonl (no perturbation suffix)

with open(MASTER_LIST, 'r') as f:
    dataset_paths = [line.strip() for line in f if line.strip()]

# Build a mapping: dataset_name -> entry count
dataset_counts = {}
for rel_path in dataset_paths:
    filename = os.path.basename(rel_path)

    # Only process English (unperturbed) datasets: those ending in %filtered.jsonl
    if not filename.endswith("%filtered.jsonl"):
        continue

    # Extract dataset name: everything before the timestamp
    # Format: {dataset_name}_{timestamp}%filtered.jsonl
    # e.g. "adjunct_island_20260101_170829%filtered.jsonl" -> "adjunct_island"
    name_part = filename.replace("%filtered.jsonl", "")
    # Remove the timestamp (last two underscore-separated numeric segments)
    match = re.match(r"^(.+?)_(\d{8}_\d{6})$", name_part)
    if match:
        dataset_name = match.group(1)
    else:
        print(f"Warning: could not parse dataset name from {filename}")
        continue

    # Count lines in the file
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        print(f"Warning: file not found: {full_path}")
        continue

    with open(full_path, 'r') as df:
        count = sum(1 for _ in df)

    dataset_counts[dataset_name] = count
    pct = (EXPECTED_TOTAL - count) / EXPECTED_TOTAL * 100
    print(f"  {dataset_name:55s}  {count:5d} entries  ({pct:.1f}% discarded)")

print(f"\nProcessed {len(dataset_counts)} English datasets.")

# ---- Step 2: Update the CSV with 'Number of minimal pairs' and '% Discarded' columns ----

with open(CSV_PATH, 'r', newline='') as f:
    reader = csv.reader(f)
    rows = list(reader)

header = rows[0]

# Add or locate the 'Number of minimal pairs' column
if "Number of minimal pairs" in header:
    count_col_idx = header.index("Number of minimal pairs")
else:
    header.append("Number of minimal pairs")
    count_col_idx = len(header) - 1
    for row in rows[1:]:
        row.append("")

# Add or locate the '% Discarded' column
if "% Discarded" in header:
    discard_col_idx = header.index("% Discarded")
else:
    header.append("% Discarded")
    discard_col_idx = len(header) - 1
    for row in rows[1:]:
        row.append("")

updated_count = 0
for row in rows[1:]:
    if len(row) < 2:
        continue
    dataset_name = row[1].strip()
    if dataset_name in dataset_counts:
        count = dataset_counts[dataset_name]
        pct_discarded = (EXPECTED_TOTAL - count) / EXPECTED_TOTAL * 100
        # Ensure row is long enough
        while len(row) <= max(count_col_idx, discard_col_idx):
            row.append("")
        row[count_col_idx] = str(count)
        row[discard_col_idx] = f"{pct_discarded:.1f}%"
        updated_count += 1

with open(CSV_PATH, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"Updated {updated_count} rows in {CSV_PATH}")
