#!/usr/bin/env python  
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path

from evaluation.evaluate import ModelComparisonEvaluator

def main():
    # Dataset path
    dataset_path = "data_generation/outputs/impossible_blimp/english_to_local_shuffle_three_for_irregular_adj.jsonl"
    
    # Model names
    model1_name = "mission-impossible-lms/no-shuffle-gpt2"
    model2_name = "mission-impossible-lms/local-shuffle-w3-gpt2"
    
    # Output directory and filename base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = Path("experiments/output") # Use Path object
    output_filename_base = output_dir_path / f"{Path(__file__).stem}_eval_{timestamp}"
    
    # Create base output directory if it doesn't exist (raw subdir will be created by evaluator)
    os.makedirs(output_dir_path, exist_ok=True)
    
    print(f"Starting evaluation experiment with:")
    print(f"- Dataset: {dataset_path}")
    print(f"- Model 1: {model1_name}")
    print(f"- Model 2: {model2_name}")
    print(f"- Output Base: {output_filename_base}") # Updated print
    
    # Initialize the evaluator
    evaluator = ModelComparisonEvaluator(
        dataset_filepath=dataset_path,
        model_name_1=model1_name,
        model_name_2=model2_name
    )
    
    # Run the evaluation with batch processing (batch_size=16 is the default)
    # Pass the string representation of the Path object
    evaluator.evaluate(output_filename=str(output_filename_base), batch_size=16)
    
    print(f"Experiment completed. Summary and raw results saved based on: {output_filename_base}")


if __name__ == "__main__":
    main()
