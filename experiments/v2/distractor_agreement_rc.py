#!/usr/bin/env python  
# -*- coding: utf-8 -*-

import os
from datetime import datetime
from pathlib import Path

from data_generation.generation_projects.impossible_blimp.v2.distractor_agreement_rc import ImpossibleDistractorAgreementRCGenerator
from evaluation.evaluate import run_batch_perturbation_evaluation

def main():
    # Dataset path
    dataset_path = "data_generation/outputs/impossible_blimp/v2/impossible_distractor_agreement_relative_clause.jsonl"
    
    # Phenomenon class/module
    class_name = ImpossibleDistractorAgreementRCGenerator
    
    # Output directory and filename base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = Path("experiments/output/v2")
    output_filename_base = output_dir_path / f"{Path(__file__).stem}_eval_{timestamp}"
    os.makedirs(output_dir_path, exist_ok=True)

    print(f"Starting batch perturbation evaluation experiment with:")
    print(f"- Dataset: {dataset_path}")
    print(f"- Phenomenon class: {class_name}")
    print(f"- Output Base: {output_filename_base}")

    # Run batch evaluation
    summary = run_batch_perturbation_evaluation(
        dataset=dataset_path,
        phenomenon_class=class_name,
        output_base=output_filename_base,
        batch_size=16,
        write_results=True,
    )
    # Optionally, you can now use summary/all_results for further analysis in Python
    print("Summary:", summary)

if __name__ == "__main__":
    main()
