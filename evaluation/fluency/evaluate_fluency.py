# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_generation.utils.impossible_utils import VALID_PERTURBATION_KEYS
import json

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Setup GPT-2 small pre-trained model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1B").to(device)
model.eval()

RESULTS_CSV = "evaluation/fluency/fluency_scores.csv"
MODEL_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"
PYTHIA_MODEL_UNIGRAM_LOGPROBS_FILEPATH = "evaluation/fluency/pile_unigram_logprobs.json"

UNIGRAM_COEFFECIENT_BETA = 0.699
LENGTH_COEFFECIENT_GAMMA = 11.59

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
    return log_sentence_probability.item()

def load_pythia_model_unigram_scores(filepath) -> dict:
    with open(filepath, 'r') as json_file:
        scores = json.load(json_file)
        return scores

in_mem_unigram_scores = load_pythia_model_unigram_scores(PYTHIA_MODEL_UNIGRAM_LOGPROBS_FILEPATH)

def get_summed_log_unigram_probability(tokens) -> float:
    sum_log_unigram_probs = 0
    total = 0
    for token in tokens:
        sum_log_unigram_probs += in_mem_unigram_scores.get(token, 0)
        if token not in in_mem_unigram_scores:
            print(f"Warning: Token '{token}' not found in unigram scores. Defaulting to 0.")
        else:
            total += 1
    
    return sum_log_unigram_probs, total
data = []
for perturbation in tqdm(VALID_PERTURBATION_KEYS): 
    file = MODEL_OUTPUT_DIR + f"{perturbation}.txt"


    # print("Computing scores for file : ", file)
    # Read the content of the file line by line
    with open(file, "r") as f:
        lines = [line.strip() for line in f]
        for line in tqdm(lines, leave=False, desc=f"Scoring {perturbation}"):
            # calculate slor_score for this line

            # remove any 🅁 and track if we did:
            hadR = False
            if "🅁" in line:
                line_cleaned = line.replace("🅁", "")
                hadR = True
            else:
                line_cleaned = line
            
            x = get_log_sentence_probability(line_cleaned, model, tokenizer, device)

            tokens = tokenizer.tokenize(line_cleaned)
            tokens = [token.replace("Ġ", " ") for token in tokens] # remove the Ġ character that indicates a space in GPT-2 tokenization, since the unigram scores are for the base token without the Ġ
            tokens[0] = tokens[0].lstrip()
            y, denom = get_summed_log_unigram_probability(tokens)
            # print(f"Line: {line}, Log Sentence Prob: {x}, Sum Log Unigram Prob: {y}, Num Tokens: {len(tokens)}")


            morcela = (x - UNIGRAM_COEFFECIENT_BETA * y + LENGTH_COEFFECIENT_GAMMA) / denom # Using MORCELA formula, which is SLOR with coeffecients 
            slor = (x - y) / denom
            # print(f"Line: {line}, Log Sentence Prob: {x:.2f}, Sum Log Unigram Prob: {y:.2f}, Num Tokens: {denom}, Score: {score:.2f}")

            if not np.isnan(morcela):
                data.append({'perturbation': perturbation, 'generation': line, 'morcela': morcela, 'generation_cleaned': line_cleaned, 'hadR': hadR, 'perplexity': np.exp(-x/len(tokens)), 'slor': slor})

df = pd.DataFrame(data)
df.to_csv(RESULTS_CSV, index=False)
# summary statistics
print(df.groupby('perturbation')['morcela'].agg(['mean', 'std', 'quantile']))
print(df.groupby('perturbation')['perplexity'].agg(['mean', 'std', 'quantile']))
print(df.groupby('perturbation')['slor'].agg(['mean', 'std', 'quantile']))
    
    # mean, std = np.mean(fluency_scores), np.std(fluency_scores)
    # # write to RESULTS_CSV - (perturbation, mean, std_dev)
    # with open(RESULTS_CSV, 'a', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow([perturbation, mean, std])

print("Computed all fluency scores")
