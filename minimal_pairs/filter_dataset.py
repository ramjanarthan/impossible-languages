import argparse
import os
import json
import jsonlines
import sys
from tqdm import tqdm
from utils.impossible_utils import PERTURBATIONS_PAIR

BATCH_SIZE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse a BLiMP dataset into an impossible dataset using a specified perturbation."
    )
    parser.add_argument(
        "base_dataset_path", type=str, help="Path to the base BLiMP dataset (.jsonl)"
    )
    return parser.parse_args()


def filter_dataset(base_dataset_path, output_path):
    shuffle_tokenizer, reverse_tokenier = (
        PERTURBATIONS_PAIR["shuffle_deterministic21"]["gpt2_tokenizer"],
        PERTURBATIONS_PAIR["reverse_full"]["gpt2_tokenizer"],
    )

    # Prepare output path
    base_dir, base_file = os.path.split(base_dataset_path)
    fname, ext = os.path.splitext(base_file)

    try:
        with open(base_dataset_path, "r") as infile, open(output_path, "w") as outfile:
            output_writer = jsonlines.Writer(outfile, flush=True)
            batch = []
            filtered_count = 0
            total_count = 0
            for line in tqdm(infile, desc="Processing", unit="sentences"):
                data = json.loads(line)
                total_count += 1

                # Apply perturbation to both good and bad sentences
                sent_good = data.get("sentence_good")
                sent_bad = data.get("sentence_bad")
                if sent_good is None or sent_bad is None:
                    raise ValueError(
                        "Input data must have 'sentence_good'/'sentence_bad' fields."
                    )

                # Check for shuffle_tokenizer parity
                length_of_good = shuffle_tokenizer.encode(sent_good)
                length_of_bad = shuffle_tokenizer.encode(sent_bad)
                if len(length_of_good) != len(length_of_bad):
                    filtered_count += 1
                    continue

                # Check for reverse_tokenier parity
                length_of_good = reverse_tokenier.encode(sent_good)
                length_of_bad = reverse_tokenier.encode(sent_bad)
                if len(length_of_good) != len(length_of_bad):
                    filtered_count += 1
                    continue

                batch.append(data)
                if len(batch) >= BATCH_SIZE:
                    output_writer.write_all(batch)
                    batch = []
            output_writer.write_all(batch)
            output_writer.close()
        print(f"Successfully wrote filtered dataset to {output_path}")
        print(f"Filtered {filtered_count} sentences out of {total_count}")
        return 0
    except Exception as e:
        print(f"Failed to process dataset: {e}")
        return 1


def main():
    args = parse_args()
    return filter_dataset(args.base_dataset_path)


if __name__ == "__main__":
    sys.exit(main())
