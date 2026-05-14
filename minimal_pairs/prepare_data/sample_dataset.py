#!/usr/bin/env python3
"""
Script to sample and compare sentences across different language versions of the BLiMP dataset.

Usage:
    python sample_dataset.py <path_to_english_dataset> [--samples N] [--seed SEED]
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional

# Valid perturbation keys from impossible_utils.py
VALID_PERTURBATION_KEYS = [
    "english",
    "shuffle_nondeterministic",
    "shuffle_deterministic21",
    "shuffle_local3",
    "shuffle_local5",
    "shuffle_local10",
    "shuffle_even_odd", 
    "reverse_partial",
    "reverse_full",
]

def load_dataset(file_path: str) -> List[Dict]:
    """Load a JSONL dataset file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def get_other_versions(base_path: str) -> Dict[str, str]:
    """
    Find all other versions of the dataset based on valid perturbation keys.
    Returns a dictionary mapping perturbation keys to file paths.
    """
    base_path = Path(base_path)
    parent_dir = base_path.parent
    base_name = base_path.stem
    
    versions = {}
    
    # Add the base English version
    if base_path.exists():
        versions["english"] = str(base_path)
    
    # Look for other versions
    for key in VALID_PERTURBATION_KEYS:
        if key == "english":
            continue
            
        # Construct possible filenames for this perturbation
        new_file_names = f"{base_name}%{key}.jsonl"
        
        path = parent_dir / new_file_names
        if path.exists() and path != base_path:
            versions[key] = str(path)
    
    return versions

def sample_examples(datasets: Dict[str, List[Dict]], num_samples: int = 5, seed: Optional[int] = None) -> List[Dict]:
    """Sample examples from the datasets."""
    if seed is not None:
        random.seed(seed)
    
    # Get the English dataset to sample from
    english_data = datasets.get("english")
    if not english_data:
        raise ValueError("English dataset not found in the provided datasets")
    
    # Sample indices
    sample_indices = random.sample(range(len(english_data)), min(num_samples, len(english_data)))
    
    results = []
    for idx in sample_indices:
        example = {
            "index": idx,
            "english": {
                "sentence_good": english_data[idx].get("sentence_good", ""),
                "sentence_bad": english_data[idx].get("sentence_bad", "")
            },
            "other_versions": {}
        }
        
        # Get corresponding examples from other versions
        for version, data in datasets.items():
            if version == "english":
                continue
                
            if idx < len(data):
                example["other_versions"][version] = {
                    "sentence_good": data[idx].get("sentence_good", ""),
                    "sentence_bad": data[idx].get("sentence_bad", "")
                }
        
        results.append(example)
    
    return results

def display_results(samples: List[Dict]):
    """Display the sampled examples in a structured format."""
    for i, sample in enumerate(samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1} (Index: {sample['index']})")
        print(f"{'='*80}")
        
        # English version
        print("\nENGLISH:")
        print(f"  Good: {sample['english']['sentence_good']}")
        print(f"  Bad:  {sample['english']['sentence_bad']}")
        
        # Other versions
        if sample['other_versions']:
            print("\nOTHER VERSIONS:")
            for version, sentences in sample['other_versions'].items():
                print(f"\n{version.upper()}:")
                print(f"  Good: {sentences['sentence_good']}")
                print(f"  Bad:  {sentences['sentence_bad']}")
        
        print("\n" + "-"*80)

def main():
    parser = argparse.ArgumentParser(description='Sample and compare sentences across different language versions of the dataset.')
    parser.add_argument('dataset_path', type=str, help='Path to the English version of the dataset')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples to display (default: 3)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Find all dataset versions
    dataset_files = get_other_versions(args.dataset_path)
    
    if not dataset_files:
        print("Error: No dataset files found.")
        return
    
    for name, path in dataset_files.items():
        print(f"  - {name}: {path}")
    
    # Load all datasets
    datasets = {}
    for name, path in dataset_files.items():
        try:
            datasets[name] = load_dataset(path)
        except Exception as e:
            print(f"  Error loading {name}: {str(e)}")
    
    # Sample and display examples
    print(f"\nSampling {args.samples} examples...")
    samples = sample_examples(datasets, num_samples=args.samples, seed=args.seed)
    display_results(samples)

if __name__ == "__main__":
    main()