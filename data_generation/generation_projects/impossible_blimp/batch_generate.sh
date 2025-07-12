#!/bin/bash

# List of datasets to process
DATASET_LIST="data_generation/generation_projects/impossible_blimp/master_dataset_list.txt"

# List of perturbations to apply
PERTURBATIONS=(
    "shuffle_nondeterministic"
    "shuffle_deterministic21"
    "shuffle_local3"
    "shuffle_local5"
    "shuffle_local10"
    "shuffle_even_odd"
    "reverse_partial"
    "reverse_full"
)

# Process each dataset
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    # Get the base name and directory of the dataset
    dataset_dir=$(dirname "$dataset")
    dataset_base=$(basename "$dataset" .jsonl)
    
    # Process each perturbation
    for perturbation in "${PERTURBATIONS[@]}"; do
        # Define the output file name pattern
        output_file="${dataset_dir}/${dataset_base}%${perturbation}.jsonl"
        
        # Check if the perturbed file already exists
        if [ -f "$output_file" ]; then
            echo "Skipping $dataset_base with perturbation $perturbation - output file already exists: $output_file"
            echo "----------------------------------------"
            continue
        fi
        
        echo "Processing $dataset_base with perturbation: $perturbation"
        
        # Run the perturbation
        python -m data_generation.generation_projects.impossible_blimp.modify_dataset "$dataset" "$perturbation"
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Successfully processed $dataset_base with $perturbation"
        else
            echo "Error processing $dataset_base with $perturbation"
            exit 1
        fi
        
        echo "----------------------------------------"
    done
done < "$DATASET_LIST"

echo "All perturbations completed successfully!"