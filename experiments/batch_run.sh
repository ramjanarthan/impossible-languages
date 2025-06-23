#!/bin/bash

base_dataset_dir="data_generation/outputs/impossible_blimp/v2"

grammatical_phenomena=(
    "adjunct_island"
    # "animate_subject_passive"
    # "anaphor_gender_agreement"
    # "anaphor_number_agreement"
    # "distractor_agreement_relative_clause"
    # "ellipsis_n_bar_1"
    # "irregular_past_participle_adjectives"
    # "principle_A_c_command"
    # "wh_questions_object_gap"
    # "wh_questions_object_gap_long_distance"
    # "wh_questions_subject_gap"
    # "wh_questions_subject_gap_long_distance"
)

MODEL_FAMILIES=(
    "english"
    # "shuffle_nondeterministic"
    # "shuffle_deterministic21" 
    # "shuffle_deterministic57"
    # "shuffle_deterministic84"
    # "shuffle_local3"
    # "shuffle_local5"
    # "shuffle_local10"
    # "shuffle_even_odd"
    # "reverse_partial"
    # "reverse_full"
)
    
# define output csv
results_csv="experiments/output/v2/results.csv"

# run experiments   
for grammatical_phenomenon in "${grammatical_phenomena[@]}"; do
    for model_family in "${MODEL_FAMILIES[@]}"; do
        echo "Running experiment for grammatical_phenomenon: $grammatical_phenomenon with model family: $model_family"

        dataset_path=""
        if [ "$model_family" == "english" ]; then
            dataset_path="$base_dataset_dir/$grammatical_phenomenon.jsonl"
        else
            dataset_path="$base_dataset_dir/$grammatical_phenomenon%$model_family.jsonl"
        fi

        python -m experiments.experiment --results_csv "$results_csv" --model_name "$model_family" --dataset "$dataset_path"
    done

    # git commit changes with message "batch run experiment -${grammatical_phenomenon}"
    git add .
    git commit -m "batch run experiment -${grammatical_phenomenon}" -m "Model families: ${MODEL_FAMILIES} \nDataset: ${dataset_path}"
    git push
done
