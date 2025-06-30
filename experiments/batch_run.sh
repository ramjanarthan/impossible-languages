#!/bin/bash

base_dataset_dir="data_generation/outputs/impossible_blimp/v2"

grammatical_phenomena=(
    # "adjunct_island_20250623_165451"
    "animate_subject_passive_20250623_165531"
    # "anaphor_gender_agreement_20250618_113511"
    # "anaphor_number_agreement_20250617_153306"
    # "distractor_agreement_relative_clause_20250618_125716"
    "ellipsis_n_bar_1_20250623_165615"
    # "irregular_past_participle_adjectives_20250618_141423"
    # "principle_A_c_command_20250623_165615"
    # "wh_questions_object_gap_20250623_165615"
    # "wh_questions_object_gap_long_distance_20250623_165615"
    # "wh_questions_subject_gap_20250623_165615"
    # "wh_questions_subject_gap_long_distance_20250623_165615"
)

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

# run experiments   
for grammatical_phenomenon in "${grammatical_phenomena[@]}"; do
    for model_family in "${MODEL_FAMILIES[@]}"; do
        echo -e "\n-- Running experiment for grammatical_phenomenon: $grammatical_phenomenon with model family: $model_family-- " >> $LOG_FILE 

        dataset_path=""
        if [ "$model_family" == "english" ]; then
            dataset_path="$base_dataset_dir/$grammatical_phenomenon.jsonl"
        else
            dataset_path="$base_dataset_dir/$grammatical_phenomenon%$model_family.jsonl"
        fi

        python -m experiments.experiment --results_csv "$results_csv" --model_name "$model_family" --dataset "$dataset_path"

        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "✓ Completed successfully" >> $LOG_FILE
        else
            echo "✗ Failed to run experiment" >> $LOG_FILE
        fi
    done

    # git commit changes with message "batch run experiment -${grammatical_phenomenon}"
    git add .
    git commit -m "batch run experiment -${grammatical_phenomenon}" -m "Model families: ${MODEL_FAMILIES} Dataset: ${dataset_path}"
    git push

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ pushed successfully" >> $LOG_FILE
    else
        echo "✗ Failed to push results" >> $LOG_FILE
    fi
done
