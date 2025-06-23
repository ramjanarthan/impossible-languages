import argparse
import os
import json
import jsonlines
import sys
from tqdm import tqdm
from data_generation.utils.impossible_utils import PERTURBATIONS, PERTURBATION_TO_HF_MODEL_NAME

BATCH_SIZE = 16

def parse_args():
    parser = argparse.ArgumentParser(description="Modify a BLiMP dataset into an impossible dataset using a specified perturbation.")
    parser.add_argument("base_dataset_path", type=str, help="Path to the base BLiMP dataset (.jsonl)")
    parser.add_argument("impossible_language_option", type=str, help="Perturbation option (must be a key in PERTURBATION_TO_HF_MODEL_NAME)")
    parser.add_argument("--output_path", type=str, default=None, help="Optional output path. Defaults to input path + '%' + perturbation id + extension.")
    return parser.parse_args()

def modify_dataset(base_dataset_path, impossible_language_option, output_path=None):
    if impossible_language_option not in PERTURBATION_TO_HF_MODEL_NAME:
        raise ValueError(f"Invalid impossible_language_option '{impossible_language_option}'. Must be one of: {list(PERTURBATION_TO_HF_MODEL_NAME.keys())}")
    perturbation = PERTURBATIONS[impossible_language_option]
    perturb_func = perturbation["perturbation_function"]
    tokenizer = perturbation["gpt2_tokenizer"]

    # Prepare output path
    base_dir, base_file = os.path.split(base_dataset_path)
    fname, ext = os.path.splitext(base_file)
    if output_path is None:
        output_path = os.path.join(base_dir, f"{fname}%{impossible_language_option}{ext}")

    try:
        with open(base_dataset_path, "r") as infile, open(output_path, "w") as outfile:
            output_writer = jsonlines.Writer(outfile, flush=True)
            batch = []
            for line in tqdm(infile, desc="Processing", unit="sentences"):
                data = json.loads(line)
                # Apply perturbation to both good and bad sentences
                sent_good = data.get("sentence_good")
                sent_bad = data.get("sentence_bad")
                if sent_good is None or sent_bad is None:
                    raise ValueError("Input data must have 'sentence_good'/'sentence_bad' fields.")
                # Apply perturbation
                impossible_good = perturb_func(sent_good)
                impossible_bad = perturb_func(sent_bad)
                # Decode if necessary (as in v1 batch code)

                impossible_good = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_good))
                impossible_bad = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), impossible_bad))
                
                # Add new fields
                data["sentence_good"] = impossible_good
                data["sentence_bad"] = impossible_bad
                batch.append(data)
                if len(batch) >= BATCH_SIZE:
                    for item in batch:
                        output_writer.write(item)
                    batch = []
            # Write any remaining items
            if batch:
                for item in batch:
                    output_writer.write(item)
            output_writer.close()
        print(f"Successfully wrote impossible dataset to {output_path}")
        return 0
    except Exception as e:
        print(f"Failed to process dataset: {e}")
        return 1

def main():
    args = parse_args()
    return modify_dataset(args.base_dataset_path, args.impossible_language_option, args.output_path)

if __name__ == "__main__":
    sys.exit(main())
