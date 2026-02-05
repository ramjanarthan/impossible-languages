# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import csv
import sys
from data_generation.utils.impossible_utils import VALID_PERTURBATION_KEYS

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup GPT-2 small pre-trained model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m").to(device)

RESULTS_CSV = "evaluation/slor/fluency_scores.csv"
MODEL_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"

def get_log_sentence_probability(line, model, tokenizer, device) -> float:
    encoded_line = tokenizer(
        line,
        return_tensors='pt',
        return_attention_mask=True
    ).to(device)

    output = model(
        encoded_line.input_ids,
        labels=encoded_line.input_ids,
    )

    loss_ce = output.loss

    log_sentence_probability = -loss_ce
    return log_sentence_probability


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
            print("Sentence: ", line)
            print("Sentence probability: ", x)
            sys.exit()
            y = get_summed_log_unigram_probability(line, model, tokenizer, device)
            denom = len(line)

            unigram_coefficient = 1 
            length_coefficient = 0
            score = (x - unigram_coefficient * y + length_coefficient) / denom # Using MORCELA formula, which is SLOR with coeffecients 

            slor_scores.append(score)

    mean, std = np.mean(slor_scores), np.std(slor_scores)
    
    # write to RESULTS_CSV - (perturbation, mean, std_dev)
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([perturbation, mean, std])

