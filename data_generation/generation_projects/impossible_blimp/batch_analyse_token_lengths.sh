#!/bin/bash

# List of datasets to process
DATASETS=(
    "data_generation/outputs/impossible_blimp/v2/adjunct_island_20250623_165451.jsonl"
    "data_generation/outputs/impossible_blimp/v2/anaphor_gender_agreement_20250618_113511.jsonl"
    "data_generation/outputs/impossible_blimp/v2/anaphor_number_agreement_20250617_153306.jsonl"
    "data_generation/outputs/impossible_blimp/v2/animate_subject_passive_20250623_165531.jsonl"
    "data_generation/outputs/impossible_blimp/v2/distractor_agreement_relative_clause_20250618_125716.jsonl"
    "data_generation/outputs/impossible_blimp/v2/ellipsis_n_bar_1_20250623_165615.jsonl"
    "data_generation/outputs/impossible_blimp/v2/irregular_past_participle_adjectives_20250618_141423.jsonl"
    "data_generation/outputs/impossible_blimp/v2/principle_A_c_command_20250623_165615.jsonl"
    "data_generation/outputs/impossible_blimp/v2/wh_questions_object_gap_20250623_165615.jsonl"
    "data_generation/outputs/impossible_blimp/v2/wh_questions_object_gap_long_distance_20250623_165615.jsonl"
    "data_generation/outputs/impossible_blimp/v2/wh_questions_subject_gap_20250623_165615.jsonl"
    "data_generation/outputs/impossible_blimp/v2/wh_questions_subject_gap_long_distance_20250623_165615.jsonl"
)

# Process each dataset
for dataset in "${DATASETS[@]}"; do
    # Get the base name and directory of the dataset
    dataset_dir=$(dirname "$dataset")
    dataset_base=$(basename "$dataset" .jsonl)
    
   
    echo "Processing $dataset_base"
        
    # Run the perturbation
    python -m data_generation.generation_projects.impossible_blimp.analyse_token_lengths "$dataset" 
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully processed $dataset_base"
    else
        echo "Error processing $dataset_base"
        exit 1
    fi
    
    echo "----------------------------------------"
done

echo "All datasets processed successfully!"