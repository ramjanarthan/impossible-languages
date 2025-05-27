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
    
    # Output directory and filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "experiments/output"
    output_filename = f"{output_dir}/anaphor_agreement_eval_{timestamp}.csv"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting evaluation experiment with:")
    print(f"- Dataset: {dataset_path}")
    print(f"- Model 1: {model1_name}")
    print(f"- Model 2: {model2_name}")
    print(f"- Output: {output_filename}")
    
    # Initialize the evaluator
    evaluator = ModelComparisonEvaluator(
        dataset_filepath=dataset_path,
        model_name_1=model1_name,
        model_name_2=model2_name
    )
    
    # Run the evaluation with batch processing (batch_size=16 is the default)
    evaluator.evaluate(output_filename=output_filename, batch_size=16)
    
    print(f"Experiment completed. Results saved to {output_filename}")


if __name__ == "__main__":
    main()
