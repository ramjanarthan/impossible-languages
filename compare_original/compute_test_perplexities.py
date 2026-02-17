import os
import gc
import json
import torch
from data_generation.utils.impossible_utils import VALID_PERTURBATION_KEYS, PERTURBATION_TO_HF_MODEL_NAME
from evaluation.perplexity import get_perplexities_from_token_ids, calculate_geometric_mean_perplexity
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---- Config ----
DEVICE = "mps"
BATCH_SIZE = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def load_test_file(file_path):
    """Read a .test file containing JSON-encoded token ID lists."""
    token_id_lists = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                token_id_lists.append(json.loads(line))
    return token_id_lists


def calculate_dataset_perplexity(model, tokenizer, dataset, device=DEVICE, batch_size=BATCH_SIZE):
    """Calculate geometric-mean perplexity over a dataset in small batches."""
    all_perplexities = []

    for i in tqdm(range(0, len(dataset), batch_size), desc="Computing perplexity"):
        batch_token_lists = dataset[i:i + batch_size]
        batch_perplexities = get_perplexities_from_token_ids(model, tokenizer, batch_token_lists, device)
        all_perplexities.extend(batch_perplexities)

        # Free MPS cache periodically to avoid system freeze
        if device == "mps":
            torch.mps.empty_cache()

    return calculate_geometric_mean_perplexity(all_perplexities)


# ---- Main loop ----
results = {}

for key in VALID_PERTURBATION_KEYS:
    test_file = os.path.join(OUTPUT_DIR, f"{key}.test")

    if not os.path.exists(test_file):
        print(f"[{key}] ⚠ File not found: {test_file} — skipping")
        continue

    # Load dataset
    dataset = load_test_file(test_file)
    print(f"\n[{key}] Loaded {len(dataset)} lines from {os.path.basename(test_file)}")

    # Load model & tokenizer
    model_id = PERTURBATION_TO_HF_MODEL_NAME[key]
    print(f"[{key}] Loading model: {model_id}")
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model.to(DEVICE)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    # Compute perplexity
    perplexity = calculate_dataset_perplexity(model, tokenizer, dataset)
    results[key] = perplexity
    print(f"[{key}] Geometric mean perplexity: {perplexity:.4f}")

    # Cleanup model to free memory before loading the next one
    del model, tokenizer
    torch.mps.empty_cache()
    gc.collect()

# ---- Summary ----
print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
for key, ppl in results.items():
    print(f"  {key:30s} → {ppl:.4f}")

# ---- Write results to file ----
results_file = os.path.join(OUTPUT_DIR, "results.txt")
with open(results_file, 'w') as f:
    f.write("RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n")
    for key, ppl in results.items():
        f.write(f"  {key:30s} → {ppl:.4f}\n")
print(f"\nResults written to {results_file}")