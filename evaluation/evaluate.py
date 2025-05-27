import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List
from evaluation.evaluation_dataset import ParallelEvaluationDatasetIterator
import math
import pandas as pd
from tqdm import tqdm

# Helper function to calculate perplexity (adapted from partial_reverse.py)
def calculate_single_sentence_perplexity(model, tokenizer, sentence: str, device, max_length=512):
    """
    Calculates the perplexity for a single sentence.
    """
    model.eval()
    with torch.no_grad():
        encodings = tokenizer(
            [sentence], # Pass as a list for batching
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)

        labels = input_ids.clone()
        labels[input_ids == tokenizer.pad_token_id] = -100

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss # Already averaged over non-masked tokens for a single sentence

        # Count non-masked tokens for this single sentence to get true average loss for PPL
        num_tokens = (labels != -100).sum().item()
        if num_tokens == 0: # Handle cases with empty or only padding tokens
            return float('inf') # Or 0, depending on desired behavior for empty sentences

        average_loss = loss.item() # outputs.loss is already the mean loss per token
        perplexity = math.exp(average_loss)
        return perplexity


def calculate_batch_perplexity(model, tokenizer, sentences: List[str], device, max_length=512):
    """
    Calculates perplexity for a batch of sentences efficiently.
    
    Args:
        model: The language model
        tokenizer: The tokenizer for the model
        sentences: List of sentences to process in a batch
        device: The device to run computation on
        max_length: Maximum token length for truncation
        
    Returns:
        List of perplexity scores corresponding to each sentence in the batch
    """
    model.eval()
    perplexities = []
    
    # Process empty batch case
    if not sentences:
        return perplexities
    
    with torch.no_grad():
        # Tokenize all sentences in one go
        encodings = tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        input_ids = encodings.input_ids.to(device)
        attention_mask = encodings.attention_mask.to(device)
        
        # Create labels (for computing loss)
        labels = input_ids.clone()
        # Mark padding tokens with -100 so they're not included in loss computation
        labels[input_ids == tokenizer.pad_token_id] = -100
        
        # Get model outputs
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        
        # Compute per-sentence loss and perplexity
        # The loss from outputs is averaged over the entire batch,
        # so we need to compute per-sentence loss manually
        
        # Reshape logits to (batch_size, seq_len, vocab_size)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        # Reshape labels to (batch_size, seq_len)
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate loss for each token position (ignoring padded positions)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape losses to match the input shape
        losses = losses.view(shift_labels.size())
        
        # Calculate loss per sentence
        for i in range(len(sentences)):
            # Get the relevant parts for this sentence
            sentence_labels = shift_labels[i]
            sentence_losses = losses[i]
            
            # Mask out padding tokens
            mask = sentence_labels != -100
            masked_losses = sentence_losses[mask]
            
            if len(masked_losses) == 0:
                # Handle empty or completely padded sentences
                perplexities.append(float('inf'))
            else:
                # Calculate average loss and perplexity
                avg_loss = masked_losses.mean().item()
                perplexity = math.exp(avg_loss)
                perplexities.append(perplexity)
    
    return perplexities


class ParallelDataLoader:
    """
    A utility class to efficiently load and batch parallel evaluation data.
    """
    def __init__(self, filepath: str, batch_size: int = 16):
        self.filepath = filepath
        self.batch_size = batch_size
    
    def __iter__(self):
        """
        Yields batches of ParallelEvaluationDataItem objects
        """
        items_batch = []
        for item in ParallelEvaluationDatasetIterator(self.filepath):
            items_batch.append(item)
            
            if len(items_batch) >= self.batch_size:
                yield items_batch
                items_batch = []
        
        # Yield any remaining items
        if items_batch:
            yield items_batch

class ModelComparisonEvaluator:
    """
    Evaluates and compares two language models on a parallel dataset.
    For each pair in the dataset, it calculates perplexity for dataset A with model 1
    and dataset B with model 2, and then determines which model correctly
    identified the grammatical sentence as having lower perplexity.
    """
    def __init__(self, dataset_filepath: str, model_name_1: str, model_name_2: str):
        """
        Initializes the evaluator with the dataset path and two model names.

        Args:
            dataset_filepath (str): Path to the parallel evaluation dataset (JSON Lines).
            model_name_1 (str): Name of the first model (e.g., "gpt2").
            model_name_2 (str): Name of the second model (e.g., "mission-impossible-lms/partial-reverse-gpt2").
        """
        self.dataset_filepath = dataset_filepath
        self.model_name_1 = model_name_1
        self.model_name_2 = model_name_2

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models and tokenizers
        print(f"Loading Model 1: {self.model_name_1}")
        self.model1, self.tokenizer1 = self._load_model_and_tokenizer(self.model_name_1)
        print(f"Loading Model 2: {self.model_name_2}")
        self.model2, self.tokenizer2 = self._load_model_and_tokenizer(self.model_name_2)

        # Initialize counters
        self.total_pairs = 0
        self.model1_correct_count = 0
        self.model2_correct_count = 0
        self.both_correct_count = 0
        self.neither_correct_count = 0
        self.model1_only_correct_count = 0
        self.model2_only_correct_count = 0

        # Store detailed results for tabular output
        self.results_data = []

    def _load_model_and_tokenizer(self, model_name):
        """
        Helper method to load a model and its tokenizer.
        """
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.to(self.device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def evaluate(self, output_filename: str = "evaluation_results.csv", batch_size: int = 16):
        """
        Performs the evaluation by iterating through the dataset in batches, calculating
        perplexities, comparing results, and generating a summary report.

        Args:
            output_filename (str): Name of the CSV file to write results to.
            batch_size (int): Number of sentence pairs to process in a single batch.
        """
        data_loader = ParallelDataLoader(self.dataset_filepath, batch_size=batch_size)
        print(f"Starting evaluation on {self.dataset_filepath} with batch size {batch_size}...")

        for batch_items in tqdm(data_loader, desc="Processing dataset batches"):
            self.total_pairs += len(batch_items)
            
            # Extract sentences for batch processing
            sentences_A_good = []
            sentences_A_bad = []
            sentences_B_good = []
            sentences_B_bad = []
            item_indices = []
            
            for item in batch_items:
                sentences_A_good.append(item.dataset_A_grammatical)
                sentences_A_bad.append(item.dataset_A_ungrammatical)
                sentences_B_good.append(item.dataset_B_grammatical)
                sentences_B_bad.append(item.dataset_B_ungrammatical)
                item_indices.append(getattr(item, "pairID", len(self.results_data) + len(item_indices)))
            
            # Calculate perplexities in batches for efficiency
            ppls_A_good_m1 = calculate_batch_perplexity(self.model1, self.tokenizer1, sentences_A_good, self.device)
            ppls_A_bad_m1 = calculate_batch_perplexity(self.model1, self.tokenizer1, sentences_A_bad, self.device)
            ppls_B_good_m2 = calculate_batch_perplexity(self.model2, self.tokenizer2, sentences_B_good, self.device)
            ppls_B_bad_m2 = calculate_batch_perplexity(self.model2, self.tokenizer2, sentences_B_bad, self.device)
            
            # Process results for each item in the batch
            for i, item in enumerate(batch_items):
                # Get the perplexity scores for this item
                ppl_A_good_m1 = ppls_A_good_m1[i]
                ppl_A_bad_m1 = ppls_A_bad_m1[i]
                ppl_B_good_m2 = ppls_B_good_m2[i]
                ppl_B_bad_m2 = ppls_B_bad_m2[i]
                
                # Determine if models correctly preferred the grammatical sentence
                model1_is_correct = (ppl_A_good_m1 < ppl_A_bad_m1)
                model2_is_correct = (ppl_B_good_m2 < ppl_B_bad_m2)
                
                # Update counters
                if model1_is_correct:
                    self.model1_correct_count += 1
                if model2_is_correct:
                    self.model2_correct_count += 1
                
                if model1_is_correct and model2_is_correct:
                    self.both_correct_count += 1
                elif not model1_is_correct and not model2_is_correct:
                    self.neither_correct_count += 1
                elif model1_is_correct and not model2_is_correct:
                    self.model1_only_correct_count += 1
                elif not model1_is_correct and model2_is_correct:
                    self.model2_only_correct_count += 1
                
                # Store results for tabular output
                self.results_data.append({
                    "pairID": getattr(item, "pairID", len(self.results_data)),
                    "field": getattr(item, "field", "N/A"),
                    "linguistics_term": getattr(item, "linguistics_term", "N/A"),
                    "dataset_A_grammatical": item.dataset_A_grammatical,
                    "dataset_A_ungrammatical": item.dataset_A_ungrammatical,
                    f"PPL_{self.model_name_1}_A_Good": ppl_A_good_m1,
                    f"PPL_{self.model_name_1}_A_Bad": ppl_A_bad_m1,
                    f"{self.model_name_1}_Correct": model1_is_correct,
                    "dataset_B_grammatical": item.dataset_B_grammatical,
                    "dataset_B_ungrammatical": item.dataset_B_ungrammatical,
                    f"PPL_{self.model_name_2}_B_Good": ppl_B_good_m2,
                    f"PPL_{self.model_name_2}_B_Bad": ppl_B_bad_m2,
                    f"{self.model_name_2}_Correct": model2_is_correct,
                    "Both_Correct": (model1_is_correct and model2_is_correct),
                    "Model1_Only_Correct": (model1_is_correct and not model2_is_correct),
                    "Model2_Only_Correct": (not model1_is_correct and model2_is_correct),
                    "Neither_Correct": (not model1_is_correct and not model2_is_correct)
                })

        self._present_results(output_filename)

    def _present_results(self, output_filename: str):
        """
        Calculates accuracies and presents the results in a tabular form,
        writing to the specified file.
        """
        if self.total_pairs == 0:
            print("No data pairs processed. Cannot present results.")
            return

        # Calculate overall accuracies
        accuracy_m1 = (self.model1_correct_count / self.total_pairs) * 100
        accuracy_m2 = (self.model2_correct_count / self.total_pairs) * 100

        print("\n--- Evaluation Summary ---")
        print(f"Total Parallel Pairs Processed: {self.total_pairs}")
        print(f"\nAccuracy for Model 1 ({self.model_name_1}) on Dataset A: {accuracy_m1:.2f}%")
        print(f"Accuracy for Model 2 ({self.model_name_2}) on Dataset B: {accuracy_m2:.2f}%")

        print("\nComparison Counts:")
        print(f"  Both Models Correct: {self.both_correct_count} ({self.both_correct_count / self.total_pairs * 100:.2f}%)")
        print(f"  {self.model_name_1} Only Correct: {self.model1_only_correct_count} ({self.model1_only_correct_count / self.total_pairs * 100:.2f}%)")
        print(f"  {self.model_name_2} Only Correct: {self.model2_only_correct_count} ({self.model2_only_correct_count / self.total_pairs * 100:.2f}%)")
        print(f"  Neither Model Correct: {self.neither_correct_count} ({self.neither_correct_count / self.total_pairs * 100:.2f}%)")

        # Create DataFrame and write to CSV
        results_df = pd.DataFrame(self.results_data)
        results_df.to_csv(output_filename, index=False)
        print(f"\nDetailed results saved to {output_filename}")