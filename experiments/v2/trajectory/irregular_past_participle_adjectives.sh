#!/bin/bash

# Base command to run the experiment
BASE_CMD="python -m experiments.experiment"

# Model families to test
MODEL_FAMILIES=(
    "english"
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

DATASET_BASE_PATH="data_generation/outputs/impossible_blimp/v2/irregular_past_participle_adjectives_20250618_141423"

# Create a log file
LOG_FILE="experiments/v2/trajectory/irregular_past_participle_adjectives.log"

# Create a results file
RESULTS_FILE="experiments/output/v2/trajectory/irregular_past_participle_adjectives_results.csv"

# Experiment name
EXPERIMENT_NAME="irregular_past_participle_adjectives"

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
            --results_csv "$RESULTS_FILE" \
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


# Git stage all changes
git add .
# Commit changes with a message
git commit -m "trajectory experiment -${EXPERIMENT_NAME}" -m "Model families: ${MODEL_FAMILIES}" -m "Checkpoints: ${CHECKPOINTS}"

git push
