import argparse
import importlib
import json
import os
from pathlib import Path
import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from evaluation.perplexity import get_sentence_log_probabilities
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME

class PhenomenonEvaluator:
    """
    Enhanced evaluator that can dynamically load and evaluate grammatical phenomena.
    
    This class provides a clean interface for evaluating any phenomenon class
    that follows the standard interface (has PERTURBATION_KEYS_FOR_EVALUATION).
    """
    
    def __init__(self, dataset_path, phenomenon_class, output_base, batch_size=16):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the dataset JSONL file
            phenomenon_class: Either a class object or tuple of (class_name, module_path)
            output_base: Base path for output files
            batch_size: Batch size for model evaluation
        """
        self.dataset_path = dataset_path
        self.output_base = output_base
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Load data and phenomenon class
        self.data = self._load_dataset(self.dataset_path)
        self.phenomenon_class = phenomenon_class
        
        # Get perturbation keys from the class
        self.perturbation_keys = self._get_perturbation_keys()
        
        # Storage for results
        self.summary = []
        self.all_results = []
        
        print(f"Loaded {len(self.data)} examples")
        print(f"Phenomenon: {self.phenomenon_class.__name__}")
        print(f"Perturbation keys: {self.perturbation_keys}")
    
    def _get_perturbation_keys(self):
        """Get perturbation keys from the phenomenon class."""
        if hasattr(self.phenomenon_class, 'PERTURBATION_KEYS_FOR_EVALUATION'):
            return self.phenomenon_class.PERTURBATION_KEYS_FOR_EVALUATION
        else:
            raise AttributeError(f"Class {self.phenomenon_class.__name__} must have PERTURBATION_KEYS_FOR_EVALUATION attribute")

    @staticmethod
    def _load_model_and_tokenizer(model_name, device):
        """Helper method to load a model and its tokenizer."""
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _load_dataset(self, dataset_path):
        """Load dataset from JSONL file."""
        data = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _evaluate_perturbation(self, sentences_good, sentences_bad, model_name, batch_size=16):
        """Evaluate a single perturbation with the specified model."""
        model, tokenizer = self._load_model_and_tokenizer(model_name, self.device)
        
        good_logprobs = []
        bad_logprobs = []
        
        for i in range(0, len(sentences_good), batch_size):
            batch_good = sentences_good[i:i+batch_size]
            batch_bad = sentences_bad[i:i+batch_size]
            good_logprobs.extend(get_sentence_log_probabilities(model, tokenizer, batch_good, device=self.device))
            bad_logprobs.extend(get_sentence_log_probabilities(model, tokenizer, batch_bad, device=self.device))
        
        # Calculate metrics
        correct = sum(g > b for g, b in zip(good_logprobs, bad_logprobs))
        accuracy = correct / len(good_logprobs) if good_logprobs else 0
        
        # Perplexity for good and bad sentences separately
        mean_neg_logprob_good = -sum(good_logprobs) / len(good_logprobs) if good_logprobs else 0
        mean_neg_logprob_bad = -sum(bad_logprobs) / len(bad_logprobs) if bad_logprobs else 0
        perplexity_good = torch.exp(torch.tensor(mean_neg_logprob_good)).item() if good_logprobs else float('nan')
        perplexity_bad = torch.exp(torch.tensor(mean_neg_logprob_bad)).item() if bad_logprobs else float('nan')
        
        return accuracy, perplexity_good, perplexity_bad, good_logprobs, bad_logprobs

    def evaluate(self):
        """Run evaluation on all perturbations."""
        for key in self.perturbation_keys:
            model_name = PERTURBATION_TO_HF_MODEL_NAME.get(key)
            if model_name is None:
                print(f"Warning: Skipping unknown perturbation key: {key}")
                continue
                
            print(f"Evaluating perturbation '{key}' with model '{model_name}'")
            
            # Extract sentences for this perturbation
            sentences_good = []
            sentences_bad = []
            
            for ex in self.data:
                good = ex.get(f"sentence_good_{key}")
                bad = ex.get(f"sentence_bad_{key}")
                if good is not None and bad is not None:
                    sentences_good.append(good)
                    sentences_bad.append(bad)
            
            if not sentences_good:
                print(f"Warning: No data found for perturbation '{key}'")
                continue
                
            # Evaluate
            accuracy, perplexity_good, perplexity_bad, good_logprobs, bad_logprobs = self._evaluate_perturbation(
                sentences_good, sentences_bad, model_name, batch_size=self.batch_size
            )
            
            # Store summary
            self.summary.append({
                "perturbation": key,
                "model": model_name,
                "accuracy": accuracy,
                "perplexity_good": perplexity_good,
                "perplexity_bad": perplexity_bad,
                "n": len(sentences_good),
            })
            
            # Store detailed results
            for i, (g, b, g_lp, b_lp) in enumerate(zip(sentences_good, sentences_bad, good_logprobs, bad_logprobs)):
                self.all_results.append({
                    "perturbation": key,
                    "model": model_name,
                    "idx": i,
                    "sentence_good": g,
                    "sentence_bad": b,
                    "good_logprob": g_lp,
                    "bad_logprob": b_lp,
                    "correct": int(g_lp > b_lp),
                    "perplexity_good": perplexity_good,
                    "perplexity_bad": perplexity_bad,
                })
        
        return self.summary

    def write_results(self):
        """Write evaluation results to files."""
        base_path = Path(self.output_base)
        summary_txt_path = base_path.with_suffix('.txt')
        raw_csv_dir = base_path.parent / "raw"
        raw_csv_path = raw_csv_dir / base_path.with_suffix('.csv').name
        os.makedirs(raw_csv_dir, exist_ok=True)

        # Write summary
        summary_lines = []
        summary_lines.append("--- Experiment Summary ---")
        summary_lines.append(f"\nDataset: {self.dataset_path}")
        summary_lines.append(f"Phenomenon Class: {self.phenomenon_class.__name__}")
        
        summary_lines.append("\n--- Evaluation Summary ---")
        for row in self.summary:
            summary_lines.append(f"Perturbation: {row['perturbation']}")
            summary_lines.append(f"  Model: {row['model']}")
            summary_lines.append(f"  Accuracy: {row['accuracy']*100:.2f}%")
            summary_lines.append(f"  Perplexity (good): {row['perplexity_good']:.2f}")
            summary_lines.append(f"  Perplexity (bad): {row['perplexity_bad']:.2f}")
            summary_lines.append(f"  N: {row['n']}")
            summary_lines.append("")
        
        summary_content = "\n".join(summary_lines)
        print("\n" + summary_content)
        
        with open(summary_txt_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"\nEvaluation summary saved to {summary_txt_path}")

        # Write detailed results
        results_df = pd.DataFrame(self.all_results)
        results_df.to_csv(raw_csv_path, index=False)
        print(f"Detailed raw results saved to {raw_csv_path}")

def run_batch_perturbation_evaluation(dataset, phenomenon_class, output_base, batch_size=16, write_results=True):
    """
    Programmatic API for batch perturbation evaluation. Returns summary.
    If write_results is True, writes summary and CSV as well.
    """
    evaluator = PhenomenonEvaluator(
        dataset_path=dataset,
        phenomenon_class=phenomenon_class,
        output_base=output_base,
        batch_size=batch_size,
    )
    summary = evaluator.evaluate()
    if write_results:
        evaluator.write_results()
    return summary

def main():
    parser = argparse.ArgumentParser(description="Evaluate all perturbation models on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset (JSONL)")
    parser.add_argument("--class_name", type=str, required=True, help="Class name for grammatical phenomenon")
    parser.add_argument("--output_base", type=str, required=True, help="Base path for output files")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluator = PhenomenonEvaluator(
        dataset_path=args.dataset,
        phenomenon_class=args.class_name,
        output_base=args.output_base,
        batch_size=args.batch_size,
    )
    evaluator.evaluate()
    evaluator.write_results()


if __name__ == "__main__":
    main()