#!/bin/bash
# Path to the master list of filtered datasets
MASTER_LIST="data_generation/generation_projects/impossible_blimp/master_dataset_list.txt"

MODEL_FAMILIES=(
    "english"
    "shuffle_nondeterministic"
    "shuffle_deterministic21" 
    "shuffle_local3"
    "shuffle_local5"
    "shuffle_local10"
    "shuffle_even_odd"
    "reverse_partial"
    "reverse_full"
)

# define output csv
results_csv="experiments/output/v2/results.csv"

# Create a log file
LOG_FILE="experiments/output/v2/batch_run.log"

echo "Starting experiments at $(date)" > $LOG_FILE

# Read each base file from master list
while IFS= read -r filtered_file || [[ -n "$filtered_file" ]]; do
    # Skip empty lines
    [[ -z "$filtered_file" ]] && continue

    # Extract grammatical phenomenon (remove dir, %filtered, .jsonl)
    filename=$(basename "$filtered_file")
    grammatical_phenomenon="${filename%%%filtered.jsonl}"

    for model_family in "${MODEL_FAMILIES[@]}"; do
        echo -e "\n-- Running experiment for grammatical_phenomenon: $grammatical_phenomenon with model family: $model_family-- " >> $LOG_FILE 

        dataset_path=""
        if [ "$model_family" == "english" ]; then
            dataset_path="$grammatical_phenomenon%filtered.jsonl"
        else
            dataset_path="$grammatical_phenomenon%filtered%$model_family.jsonl"
        fi

        # Build the full path (same directory as filtered_file)
        base_dir=$(dirname "$filtered_file")
        full_dataset_path="$base_dir/$dataset_path"

        python -m experiments.experiment --results_csv "$results_csv" --model_name "$model_family" --dataset "$full_dataset_path"

        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully" >> $LOG_FILE
        else
            echo "✗ Failed to run experiment" >> $LOG_FILE
        fi
    done

    git add .
    git commit -m "batch run experiment -${grammatical_phenomenon}" -m "Model families: ${MODEL_FAMILIES[*]} Dataset: ${filtered_file}"
    git push

    if [ $? -eq 0 ]; then
        echo "✓ pushed successfully" >> $LOG_FILE
    else
        echo "✗ Failed to push results" >> $LOG_FILE
    fi
done < "$MASTER_LIST"
