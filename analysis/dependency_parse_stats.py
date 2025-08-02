import spacy
import pandas as pd
import numpy as np
from numpy.random import default_rng
import os

from data_generation.utils.impossible_utils import PERTURBATIONS

from analysis.dependency_parse import (
    spacy_doc_to_token_dicts,
    apply_reverse_perturbation,
    apply_partial_reverse_perturbation,
    apply_windowed_shuffle_perturbation,
    apply_shuffle_deterministic_perturbation,
    apply_shuffle_nondeterministic_perturbation,
    apply_shuffle_even_odd_perturbation,
    align_tokens_with_tokenizer,
    calculate_dependency_statistics
)


def apply_perturbations():
    """Main function to run the dependency analysis."""
    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading 'en_core_web_sm' model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Define perturbations
    perturbations = {
        'original': lambda tokens, seed: tokens,
        'reverse': lambda tokens, seed: apply_reverse_perturbation(tokens),
        'partial_reverse': lambda tokens, seed: apply_partial_reverse_perturbation(tokens, default_rng(21)),
        'shuffled_window_3': lambda tokens, seed: apply_windowed_shuffle_perturbation(tokens, 3, seed),
        'shuffled_window_5': lambda tokens, seed: apply_windowed_shuffle_perturbation(tokens, 5, seed),
        'shuffled_window_10': lambda tokens, seed: apply_windowed_shuffle_perturbation(tokens, 10, seed),
        'shuffle': lambda tokens, seed: apply_shuffle_deterministic_perturbation(tokens, seed),
        'non_deterministic_shuffle': lambda tokens, seed: apply_shuffle_nondeterministic_perturbation(tokens, default_rng(seed)),
        'odd_even_shuffle': lambda tokens, seed: apply_shuffle_even_odd_perturbation(tokens)
    }

    # Read sample sentences
    sentences_file = 'analysis/sample_sentences.txt'
    with open(sentences_file, 'r') as f:
        sentences = [line.strip() for line in f if line.strip()]

    all_stats = []
    seed = 0

    # Process each sentence
    for i, sentence in enumerate(sentences):
        print(f"Processing sentence {i+1}/{len(sentences)}: '{sentence}'")
        doc = nlp(sentence)
        original_tokens = spacy_doc_to_token_dicts(doc)

        tokenizer = PERTURBATIONS["shuffle_control"]["gpt2_tokenizer"]
        aligned_tokens = align_tokens_with_tokenizer(sentence, doc, tokenizer)

        for pert_name, pert_func in perturbations.items():
            # Apply perturbation
            perturbed_tokens = pert_func(aligned_tokens, seed + i)

            # Calculate statistics
            stats = calculate_dependency_statistics(perturbed_tokens)
            stats['sentence'] = sentence
            stats['perturbation'] = pert_name
            all_stats.append(stats)

     # Create DataFrame and save to CSV
    df = pd.DataFrame(all_stats)
    output_dir = 'analysis/output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dep_stats.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

def aggregate_stats(filepath):
     # --- Aggregate and Print Statistics ---
    print("\n--- Aggregated Statistics ---")
    
    # Group by perturbation type
    df = pd.read_csv(filepath)
    grouped = df.groupby('perturbation')

    # Calculate aggregations
    agg_stats = {}
    for name, group in grouped:
        agg_stats[name] = {
            'avg_total_dep_distance': group['total_dependency_distance'].mean(),
            'avg_norm_dep_distance': group['normalized_dependency_distance'].mean(),
            # 'avg_same_word_token_distances': group['same_word_token_distances'].mean(),
            'avg_crossing_deps': group['crossing_dependencies_count'].mean(),
            'proportion_projective': group['is_projective'].mean(), # For bool, mean is proportion of True
            'num_sentences': len(group)
        }

    # Print aggregated stats in a readable format
    for pert_name, stats in agg_stats.items():
        print(f"\n--- {pert_name.replace('_', ' ').title()} ---")
        print(f"  Number of Sentences: {stats['num_sentences']}")
        print(f"  Avg. Total Dependency Distance: {stats['avg_total_dep_distance']:.2f}")
        print(f"  Avg. Normalized Dependency Distance: {stats['avg_norm_dep_distance']:.2f}")
        print(f"  Avg. Crossing Dependencies: {stats['avg_crossing_deps']:.2f}")
        print(f"  Proportion of Projective Sentences: {stats['proportion_projective']:.2%}")


def main():
    apply_perturbations()
    aggregate_stats('analysis/output/dep_stats.csv')
   
if __name__ == "__main__":
    main()
