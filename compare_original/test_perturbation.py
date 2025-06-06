from data_generation.utils.impossible_utils import PERTURBATIONS
from evaluation.evaluate import ModelComparisonEvaluator
from evaluation.perplexity import get_perplexities, calculate_geometric_mean_perplexity
from tqdm import tqdm
from numpy.random import default_rng

file  = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/small_test/special.test"

with open(file, "r") as f:
    lines = f.readlines()

# Set parameters
FILE_SAMPLE_SIZE = 1000
RANDOM_SEED = 42
rng = default_rng(RANDOM_SEED)

# Path to the gutenberg.test file
# Adjust this path as needed for your environment
gutenberg_file = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/gutenberg.test"
bnc_file = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/bnc_spoken.test"
childes_file = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/childes.test"
switchboard_file = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/switchboard.test"
subtitles_file = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/open_subtitles.test"

def load_dataset(file_path=gutenberg_file):
  print(f"Sampling from {file_path}...")

  with open(file_path, 'r') as f:
    file_lines = [line.strip() for line in f.readlines() if line.strip()]

  print(f"Number of sentences in file: {len(file_lines)}")

  # Sample with replacement
  sample_indices = rng.choice(
      list(range(len(file_lines))), FILE_SAMPLE_SIZE, replace=True)
  return [file_lines[i] for i in sample_indices]

# gutenberg_dataset = load_dataset(gutenberg_file)
bnc_dataset = load_dataset(bnc_file)
# childes_dataset = load_dataset(childes_file)
# switchboard_dataset = load_dataset(switchboard_file)
# subtitles_dataset = load_dataset(subtitles_file)

datasets = [bnc_dataset]

local_shuffled_datasets = []
perturbation = PERTURBATIONS["shuffle_local3"]

def prepare_local_shuffle_dataset(dataset, perturbation):
    """Prepare the dataset by applying the perturbation"""
    perturbed_texts = []
    original_texts = []

    print("Applying local shuffle perturbation to sentences...")
    for i, text in enumerate(tqdm(dataset)):
        original_texts.append(text)

        # Apply perturbation with consistent seed per sentence for reproducibility
        perturbed_text_ids = perturbation["perturbation_function"](text)
        perturbed_text = "".join(map(lambda x: perturbation["gpt2_tokenizer"].decode(x), perturbed_text_ids))
        perturbed_texts.append(perturbed_text)

    return original_texts, perturbed_texts

for dataset in datasets:
  _, shuffled = prepare_local_shuffle_dataset(dataset, perturbation)
  local_shuffled_datasets.append(shuffled)

model, tokenizer = ModelComparisonEvaluator.load_model_and_tokenizer("mission-impossible-lms/local-shuffle-w3-gpt2", "mps")

def calculate_dataset_perplexity(model, tokenizer, tokens, batch_size=16, max_length=512):
    """Calculate perplexity for entire dataset"""
    all_perplexities = []

    for i in tqdm(range(0, len(tokens), batch_size), desc="Computing perplexity"):
        token_lists = tokens[i:i+batch_size]

        batch_perplexities = get_perplexities(model, tokenizer, token_lists, "mps", max_length)
        all_perplexities.extend(batch_perplexities)

    return calculate_geometric_mean_perplexity(all_perplexities)

local_shuffled_perplexities = []
for dataset in local_shuffled_datasets:
  local_shuffled_perplexities.append(calculate_dataset_perplexity(model, tokenizer, dataset))

print(f"local_shuffled_perplexities : {local_shuffled_perplexities}")