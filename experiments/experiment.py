import argparse
import sys
from evaluation.evaluate import Evaluator
from data_generation.utils.impossible_utils import IMPOSSIBLE_MODEL_CHECKPOINTS
from experiments.results import ensure_results_csv_exists, MODEL_AND_LANGUAGE_OPTIONS, DEFAULT_MODEL_CHECKPOINT

def main():
    parser = argparse.ArgumentParser(description="Run a model evaluation experiment.")
    parser.add_argument('--results_csv', type=str, required=True, help="Path to results CSV file")
    parser.add_argument('--model_name', type=str, required=True, help=f"Model name (options: {', '.join(MODEL_AND_LANGUAGE_OPTIONS)})")
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_MODEL_CHECKPOINT, help=f"Checkpoint (options: {', '.join(IMPOSSIBLE_MODEL_CHECKPOINTS)})")
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset (JSONL)")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    if args.model_name not in MODEL_AND_LANGUAGE_OPTIONS:
        print(f"ERROR: Invalid model name '{args.model_name}'.\nValid options are: {', '.join(MODEL_AND_LANGUAGE_OPTIONS)}")
        sys.exit(1)

    if args.checkpoint not in IMPOSSIBLE_MODEL_CHECKPOINTS:
        print(f"ERROR: Invalid checkpoint '{args.checkpoint}'.\nValid options are: {', '.join(IMPOSSIBLE_MODEL_CHECKPOINTS)}")
        sys.exit(1)

    ensure_results_csv_exists()

    try:
        evaluator = Evaluator(dataset_path=args.dataset, model_name=args.model_name, checkpoint=args.checkpoint, batch_size=args.batch_size, results_csv=args.results_csv)
        results = evaluator.evaluate()
        print("\nExperiment completed successfully.")
        print("--- Summary ---")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Geometric Mean Perplexity (Good): {results['perplexity_good']:.4f}")
        print(f"Geometric Mean Perplexity (Bad): {results['perplexity_bad']:.4f}")
    except Exception as e:
        print(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
