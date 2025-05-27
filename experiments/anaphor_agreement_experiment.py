#!/usr/bin/env python  
# -*- coding: utf-8 -*-

"""
First experiment using the ModelComparisonEvaluator to evaluate
no-shuffle vs. local-shuffle models on anaphor agreement gender task.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to sys.path to allow imports from sibling directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from evaluation.evaluate import ModelComparisonEvaluator

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    # Dataset path
    dataset_path = "data_generation/outputs/impossible_blimp/english_to_local_shuffle_three_for_anaphor_agreement_gender.jsonl"
    
    # Model names
    model1_name = "mission-impossible-lms/no-shuffle-gpt2"
    model2_name = "mission-impossible-lms/local-shuffle-w3-gpt2"
    
    # Output directory and filename base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = Path("experiments/output") # Use Path object
    output_filename_base = output_dir_path / f"anaphor_agreement_eval_{timestamp}"
    
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
