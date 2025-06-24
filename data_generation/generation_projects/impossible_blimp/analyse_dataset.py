import argparse
import sys
import numpy as np
from evaluation.evaluation_dataset import EvaluationDatasetIterator


def compute_stats(lengths):
    if not lengths:
        return None
    arr = np.array(lengths)
    return {
        'count': len(arr),
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'min': int(np.min(arr)),
        'max': int(np.max(arr))
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics for sentence_good lengths.")
    parser.add_argument("filepath", type=str, help="Path to the .jsonl dataset file")
    args = parser.parse_args()

    lengths = []
    try:
        for item in EvaluationDatasetIterator(args.filepath):
            if hasattr(item, 'sentence_good'):
                length = len(str(item.sentence_good).split())
                lengths.append(length)
    except Exception as e:
        print(f"Error reading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    stats = compute_stats(lengths)
    if stats is None:
        print("No valid sentence_good entries found.")
    else:
        print(f"Statistics for 'sentence_good' lengths in {args.filepath}:")
        print(f"  Count:  {stats['count']}")
        print(f"  Mean:   {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Min:    {stats['min']}")
        print(f"  Max:    {stats['max']}")

if __name__ == "__main__":
    main()
