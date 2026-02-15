import os
from data_generation.utils.impossible_utils import VALID_PERTURBATION_KEYS, PERTURBATIONS
from tqdm import tqdm
from numpy.random import default_rng

# Set parameters
FILE_SAMPLE_SIZE = 200
RANDOM_SEED = 42
rng = default_rng(RANDOM_SEED)

# Paths
BABYLM_TEST_DIR = os.path.join(os.path.dirname(__file__), "babylm_test")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Step 1: Sample and combine dataset ----

def load_dataset(file_path, sample_size, rng):
    """Load a file and sample `sample_size` lines (with replacement)."""
    with open(file_path, 'r') as f:
        file_lines = [line.strip() for line in f if line.strip()]

    sample_indices = rng.choice(len(file_lines), sample_size, replace=True)
    return [file_lines[i] for i in sample_indices]


# Discover all .test files in babylm_test/
test_files = sorted([
    os.path.join(BABYLM_TEST_DIR, f)
    for f in os.listdir(BABYLM_TEST_DIR)
    if f.endswith(".test")
])

combined_dataset = []
for file_path in test_files:
    sampled = load_dataset(file_path, FILE_SAMPLE_SIZE, rng)
    print(f"  Sampled {len(sampled)} lines from {os.path.basename(file_path)}")
    combined_dataset.extend(sampled)

print(f"Combined dataset size: {len(combined_dataset)}")

# ---- Step 2: Apply perturbations and write output files ----

def apply_perturbation(dataset, perturbation):
    """Apply a perturbation to every line and return decoded text."""
    perturbed_texts = []
    tokenizer = perturbation["gpt2_tokenizer"]

    for text in tqdm(dataset, desc="Perturbing"):
        token_ids = perturbation["perturbation_function"](text)
        perturbed_text = "".join(
            tokenizer.decode(tid) for tid in token_ids
        )
        perturbed_texts.append(perturbed_text)

    return perturbed_texts


for key in VALID_PERTURBATION_KEYS:
    output_path = os.path.join(OUTPUT_DIR, f"{key}.test")

    if key == "english":
        # No perturbation – write the original combined dataset
        dataset = combined_dataset
        print(f"[{key}] Writing original (unperturbed) dataset...")
    else:
        perturbation = PERTURBATIONS[key]
        print(f"[{key}] Applying perturbation...")
        dataset = apply_perturbation(combined_dataset, perturbation)

    with open(output_path, 'w') as f:
        for line in dataset:
            f.write(line + "\n")

    print(f"[{key}] Wrote {len(dataset)} lines → {output_path}")

print("\nDone! All perturbed test sets written to", OUTPUT_DIR)
