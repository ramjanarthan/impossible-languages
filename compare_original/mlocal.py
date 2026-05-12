import gzip
import math
import kenlm
import argparse
import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count


def load_data(train_path: Path) -> list:
    """
    Load sentences from specified files for train, validation, and test.
    Each line is treated as a sentence.
    Returns a list of all sentences.
    """
    all_sentences = []

    for path in [train_path]:
        with open(path, "r") as f:
            sentences = f.read().splitlines()
            all_sentences.extend(sentences)

    return all_sentences


def calculate_entropy(model, text):
    log_prob_sum = 0
    word_count = 0

    for line in text:
        log_prob_sum += model.score(line) * math.log2(10)
        word_count += len(line.split()) + 1

    return -1 * (log_prob_sum / word_count)


def caculate_mlocal_entropy(model, text, n: int):
    total_local_entropy = 0
    denominator = 0

    for line in text:
        scores = list(model.full_scores(line))
        valid_scores = scores[n - 1 :]
        if len(valid_scores) == 0:
            continue
        assert len(valid_scores) == len(line.split()) - n + 2, (
            f"{len(valid_scores)} != {len(line.split()) - n + 2}"
        )
        for prob, _, _ in valid_scores:
            local_entropy = -prob * math.log2(10)
            total_local_entropy += local_entropy
            denominator += 1

    return total_local_entropy / denominator if denominator > 0 else float("inf")


def main():
    for file in Path("output_train").glob("*.arpa"):
        print(f"Processing {file}...")
        sentences = load_data("output_train/" + file.stem + ".train")
        print(f"Loaded {len(sentences)} sentences from {file}")
        model = kenlm.Model(str(file))
        mlocal = caculate_mlocal_entropy(model, sentences, 4)
        print(file.stem, mlocal)


if __name__ == "__main__":
    main()