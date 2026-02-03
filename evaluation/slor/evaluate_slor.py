# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import csv
from data_generation.utils.impossible_utils import VALID_PERTURBATION_KEYS

# Setup GPT-2 small pre-trained model
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
device = "mps"

RESULTS_CSV = "evaluation/slor/slor_scores.csv"
MODEL_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"

def get_log_sentence_probability(line, model, tokenizer, device) -> float:
    return 1



in_mem_cache = {}

def get_summed_log_unigram_probability(line, model, tokenizer, device) -> float:
    tokens = tokenizer.encode(line, return_tensors='pt').to(device)[0]
    
    return 1

for perturbation in VALID_PERTURBATION_KEYS:
    file = MODEL_OUTPUT_DIR + f"{perturbation}.txt"

    slor_scores = []
    # Read the content of the file line by line
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
        for line in lines:
            # calculate slor_score for this line
            x = get_log_sentence_probability(line, model, tokenizer, device)
            y = get_summed_log_unigram_probability(line, model, tokenizer, device)
            denom = len(line)
            score = (x - y) / denom

            slor_scores.append(score)

    mean, std = np.mean(slor_scores), np.std(slor_scores)
    
    # write to RESULTS_CSV - (perturbation, mean, std_dev)
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([perturbation, mean, std])

