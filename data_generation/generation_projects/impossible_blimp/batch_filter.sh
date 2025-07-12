#!/bin/bash

# Path to the dataset list
DATASET_LIST="/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages/data_generation/generation_projects/impossible_blimp/master_dataset_list.txt"

echo "Datasets:"
cat "$DATASET_LIST"
echo

# Process each dataset line by line
while IFS= read -r dataset || [[ -n "$dataset" ]]; do
    # Skip empty lines
    [[ -z "$dataset" ]] && continue

    echo "Filtering $dataset"
    python -m data_generation.generation_projects.impossible_blimp.filter_dataset "$dataset"

    if [ $? -eq 0 ]; then
        echo "Successfully filtered $dataset"
    else
        echo "Error filtering $dataset"
        exit 1
    fi

    echo "----------------------------------------"
done < "$DATASET_LIST"

echo "Filtering completed successfully!"