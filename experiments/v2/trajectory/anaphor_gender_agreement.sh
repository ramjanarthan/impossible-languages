#!/bin/bash

# Base command to run the experiment
BASE_CMD="python -m experiments.experiment"

# Model families to test
MODEL_FAMILIES=(
    # "english"
    "shuffle_nondeterministic"
    "shuffle_deterministic21"
    "shuffle_local3"
    "reverse_full"
)

# Checkpoints to test
CHECKPOINTS=(
    "checkpoint-500"
    "checkpoint-1000"
    "checkpoint-1500"
    "checkpoint-2000"
    "checkpoint-2500"
    "checkpoint-3000"
)

DATASET_BASE_PATH="data_generation/outputs/impossible_blimp/v2/anaphor_gender_agreement_20250618_113511"

# Create a log file
LOG_FILE="experiments/v2/trajectory/anaphor_gender_agreement.log"

echo "Starting experiments at $(date)" > $LOG_FILE

# Run experiments for each model family and checkpoint
for model_family in "${MODEL_FAMILIES[@]}"; do
    echo -e "\n=== Testing model family: $model_family ===" >> $LOG_FILE

    if [ "$model_family" == "english" ]; then
        dataset_path="${DATASET_BASE_PATH}.jsonl"
    else
        dataset_path="${DATASET_BASE_PATH}%${model_family}.jsonl"
    fi
    
    for checkpoint in "${CHECKPOINTS[@]}"; do
        echo "  - Running checkpoint: $checkpoint" >> $LOG_FILE
        
        # Run the experiment command with the current model family and checkpoint
        $BASE_CMD \
            --model_name "$model_family" \
            --checkpoint "$checkpoint" \
            --dataset "$dataset_path" \
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "    ✓ Completed successfully"  >> $LOG_FILE
        else
            echo "    ✗ Failed to run experiment" >> $LOG_FILE
        fi
    done
done

echo -e "\nAll experiments completed at $(date)" >> $LOG_FILE

# Make sure git branch is 'teaching-cluster'
git checkout teaching-cluster

if [ $? -ne 0 ]; then
    echo "Failed to switch to 'teaching-cluster' branch. Exiting."
    exit 1
fi
# Git stage all changes
git add .
# Commit changes with a message
git commit -m "Run trajectory experiments for anaphor gender agreement"

# Push changes to the remote repository
git push origin teaching-cluster