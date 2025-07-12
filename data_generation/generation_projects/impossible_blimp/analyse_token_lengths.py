import argparse
import os
import json
import jsonlines
import sys
from tqdm import tqdm
from data_generation.utils.impossible_utils import PERTURBATIONS

BATCH_SIZE = 16

def parse_args():
    parser = argparse.ArgumentParser(description="Analyse a BLiMP dataset into an impossible dataset using a specified perturbation.")
    parser.add_argument("base_dataset_path", type=str, help="Path to the base BLiMP dataset (.jsonl)")
    return parser.parse_args()

def analyse_token_lengths(base_dataset_path):
    for option in ["shuffle_control", "reverse_full"]:
        tokenizer = PERTURBATIONS[option]["gpt2_tokenizer"]
        try:
            with open(base_dataset_path, "r") as infile:
                number_equal = 0
                total = 0
                avg_diff = 0
                for line in tqdm(infile, desc="Processing", unit="sentences"):
                    data = json.loads(line)
                    # Apply perturbation to both good and bad sentences
                    sent_good = data.get("sentence_good")
                    sent_bad = data.get("sentence_bad")
                    if sent_good is None or sent_bad is None:
                        raise ValueError("Input data must have 'sentence_good'/'sentence_bad' fields.")

                    length_of_good = tokenizer.encode(sent_good)   
                    length_of_bad = tokenizer.encode(sent_bad)

                    

                    # record if length of impossible_good and impossible_bad are the same as length_of_good and length_of_bad
                    if len(length_of_good) == len(length_of_bad):
                        number_equal += 1
                    
                    avg_diff += abs(len(length_of_good) - len(length_of_bad))
                    total += 1
            print(f"Result for {option} : {number_equal}/{total} \n Average difference: {avg_diff/total}")
        except Exception as e:
            print(f"Failed to process dataset: {e}")
            return 1
    return 0

def main():
    args = parse_args()
    return analyse_token_lengths(args.base_dataset_path)

if __name__ == "__main__":
    sys.exit(main())
