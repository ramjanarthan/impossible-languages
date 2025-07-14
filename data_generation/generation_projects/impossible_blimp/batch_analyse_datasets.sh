#!/bin/bash

# Path to the dataset list
DATASET_LIST="data_generation/generation_projects/impossible_blimp/master_dataset_list.txt"

# Process each dataset line by line
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    # Skip empty lines
    [[ -z "$dataset" ]] && continue

    echo "Analysing $dataset"
    python -m data_generation.generation_projects.impossible_blimp.analyse_dataset "$dataset"

    if [ $? -eq 0 ]; then
        echo "Successfully analysed $dataset"
    else
        echo "Error analysing $dataset"
        exit 1
    fi

    echo "----------------------------------------"
done < "$DATASET_LIST"

echo "Dataset analysis completed successfully!"