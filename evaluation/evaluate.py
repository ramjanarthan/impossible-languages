import argparse
import os
import re
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import csv

# Local imports
from evaluation.perplexity import get_sentence_log_probabilities, get_perplexities, calculate_geometric_mean_perplexity
from evaluation.evaluation_dataset import DataBatchLoader
from experiments.results import DEFAULT_MODEL_CHECKPOINT, append_result
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME

class Evaluator:
    """
    Evaluator for grammatical phenomena.
    Takes a dataset and a model, calculates accuracy and perplexity,
    and logs the results to a central CSV file.
    """
    def __init__(self, dataset_path: str, model_name: str, checkpoint: str, batch_size: int = 16):
        """
        Args:
            dataset_path: Path to the dataset JSONL file.
            model_name: Hugging Face model name.
            batch_size: Batch size for model evaluation.
        """
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        print(f"Using device: {self.device}")

        # convert model_name into huggingFace model
        self.hugging_face_model_name = PERTURBATION_TO_HF_MODEL_NAME[self.model_name]

        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer(self.hugging_face_model_name, self.device, self.checkpoint)

        # Parse dataset information from filename
        parsed_info = self._parse_dataset_filename(os.path.basename(self.dataset_path))
        self.grammatical_phenomenon = parsed_info['phenomenon']
        self.dataset_timestamp = parsed_info['timestamp']
        self.dataset_language = parsed_info['language']

        print(f"--- Evaluation Setup ---")
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Phenomenon: {self.grammatical_phenomenon}")
        print(f"Language: {self.dataset_language}")
        print(f"Timestamp: {self.dataset_timestamp}")
        print(f"------------------------")

    def _parse_dataset_filename(self, filename: str) -> dict:
        """
        Parses a dataset filename to extract metadata.
        Format: phenomenon_YYYYMMDD_HHMMSS%language.jsonl (language is optional)
        """
        # Strip .jsonl if present
        if filename.endswith('.jsonl'):
            filename = filename[:-6]

        pattern = r"^(?P<phenomenon>.+?)_(?P<timestamp>\d{8}_\d{6})(?:%(?P<language>.+))?$"
        match = re.match(pattern, filename)
        if not match:
            raise ValueError(f"Could not parse dataset filename: {filename}")
        
        return {
            'phenomenon': match.group('phenomenon'),
            'timestamp': match.group('timestamp'),
            'language': match.group('language') or 'english',
        }
    
    def _load_model_and_tokenizer(self, model_name: str, device: str, checkpoint: str):
        if checkpoint == DEFAULT_MODEL_CHECKPOINT:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name)
        else    :
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            model = GPT2LMHeadModel.from_pretrained(model_name, revision=checkpoint)
        model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def evaluate(self):
        """
        Runs the evaluation:
        1. Loads data in batches.
        2. Calculates log probabilities for accuracy.
        3. Calculates per-sentence perplexities.
        4. Computes geometric mean perplexity for good/bad sentences.
        5. Writes the final results to results.csv.
        Additionally, writes raw logprobs and perplexities for each sentence to a debug CSV.
        """

        batch_loader = DataBatchLoader(self.dataset_path, batch_size=self.batch_size)
        
        correct_predictions = 0
        total_sentences = 0
        all_perplexities_good = []
        all_perplexities_bad = []

        # Prepare raw output file
        raw_dir = 'experiments/output/v2/raw'
        os.makedirs(raw_dir, exist_ok=True)
        # Clean dataset name for filename
        dataset_base = os.path.basename(self.dataset_path).replace('.jsonl', '').replace('%', '_')
        filename = f"{self.model_name}_{self.checkpoint}_{dataset_base}_{self.dataset_timestamp}.csv"
        raw_path = os.path.join(raw_dir, filename)
        raw_rows = []

        print("Starting evaluation...")
        for batch in batch_loader:
            sentences_good = [item.sentence_good for item in batch]
            sentences_bad = [item.sentence_bad for item in batch]

            # 1. Calculate log probabilities for accuracy
            logprobs_good = get_sentence_log_probabilities(self.model, self.tokenizer, sentences_good, device=self.device)
            logprobs_bad = get_sentence_log_probabilities(self.model, self.tokenizer, sentences_bad, device=self.device)
            
            correct_predictions += sum(g > b for g, b in zip(logprobs_good, logprobs_bad))
            total_sentences += len(sentences_good)

            # 2. Calculate per-sentence perplexities
            perplexities_good = get_perplexities(self.model, self.tokenizer, sentences_good, device=self.device)
            perplexities_bad = get_perplexities(self.model, self.tokenizer, sentences_bad, device=self.device)

            all_perplexities_good.extend(perplexities_good)
            all_perplexities_bad.extend(perplexities_bad)

            # Pair good and bad sentences with their metrics
            for good_s, good_lp, good_ppl, bad_s, bad_lp, bad_ppl in zip(
                sentences_good, logprobs_good, perplexities_good,
                sentences_bad, logprobs_bad, perplexities_bad
            ):
                # Determine if model got it right (good sentence should have higher logprob)
                correct = good_lp > bad_lp
                raw_rows.append([
                    good_s, good_lp, good_ppl,
                    bad_s, bad_lp, bad_ppl,
                    correct
                ])

        # Write paired data to CSV
        with open(raw_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'good_sentence', 'good_logprob', 'good_perplexity',
                'bad_sentence', 'bad_logprob', 'bad_perplexity',
                'is_correct'
            ])
            writer.writerows(raw_rows)
        print(f"Paired evaluation results saved to {raw_path}")

        print(f"Evaluation finished. Processed {total_sentences} sentence pairs.")

        # 3. Calculate final metrics
        accuracy = correct_predictions / total_sentences if total_sentences > 0 else 0
        
        # 4. Geometric mean perplexity
        geo_mean_perplexity_good = calculate_geometric_mean_perplexity(all_perplexities_good)
        geo_mean_perplexity_bad = calculate_geometric_mean_perplexity(all_perplexities_bad)

        # 5. Log results to CSV
        append_result(
            model_name=self.model_name,
            checkpoint=self.checkpoint,
            grammatical_phenomenon=self.grammatical_phenomenon,
            dataset_language=self.dataset_language,
            accuracy=accuracy,
            perplexity_good=geo_mean_perplexity_good,
            perplexity_bad=geo_mean_perplexity_bad,
            dataset_path=self.dataset_path,
        )
        print("Results have been saved to experiments/output/v2/results.csv")
        
        return {
            'accuracy': accuracy,
            'perplexity_good': geo_mean_perplexity_good,
            'perplexity_bad': geo_mean_perplexity_bad,
        }

def main():
    """Command-line interface to run the evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset for a grammatical phenomenon.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset JSONL file.")
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name (e.g., 'gpt2').")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    args = parser.parse_args()

    evaluator = Evaluator(
        dataset_path=args.dataset,
        model_name=args.model_name,
        batch_size=args.batch_size
    )
    evaluator.evaluate()

if __name__ == "__main__":
    main()