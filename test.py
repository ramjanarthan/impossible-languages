from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import sys
import math
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_UNDO_PERTURBATION_KEYS, UNDO_PERTURBATIONS, PERTURBATIONS
import json

device = "mps"

# Pythia model
pythia_model_id = "EleutherAI/pythia-14M"
pythia_model, pythia_tokenizer = AutoModelForCausalLM.from_pretrained(pythia_model_id).to(device), AutoTokenizer.from_pretrained(pythia_model_id) 
pythia_model.eval()

PYTHIA_MODEL_UNIGRAM_LOGPROBS_FILEPATH = "evaluation/fluency/pile_unigram_logprobs.json"
UNIGRAM_COEFFECIENT_BETA = 0.699
LENGTH_COEFFECIENT_GAMMA = 11.59

# GPT-2 impossible model
imp_model_id = "mission-impossible-lms/no-shuffle-gpt2"
imp_model, imp_tokenizer = GPT2LMHeadModel.from_pretrained(imp_model_id).to(device), GPT2Tokenizer.from_pretrained(imp_model_id)
imp_model.eval()

token_list = [10919, 389, 345, 1972, 503, 286, 534, 474, 6475, 444, 5633]
print(imp_tokenizer.decode(token_list))

print(imp_tokenizer.decode([50256] + token_list + [50256]))
sys.exit()

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

in_mem_unigram_scores = load_pythia_model_unigram_scores(PYTHIA_MODEL_UNIGRAM_LOGPROBS_FILEPATH)

# a = "What's that you say?"
# b = "What's that yousay?"

# a = "Have you seen them?"
# b = "Have you seen them? -"

a = "Let go of it."
b = "Let let go of it."

print(a)
print(pythia_tokenizer.tokenize(a))

print(b)
print(pythia_tokenizer.tokenize(b))

# print("----")
# for str in ["say", "Ġsay", " say"]:
#     print(f"Logprob for '{str}' :", in_mem_unigram_scores[str])

print("-----")

for sent in [a, b]:
    x = get_log_sentence_probability(sent, pythia_model, pythia_tokenizer, device)
    tokens = pythia_tokenizer.tokenize(sent)

    tokens = [token.replace("Ġ", " ") for token in tokens] # remove the Ġ character that indicates a space in GPT-2 tokenization, since the unigram scores are for the base token without the Ġ
    tokens[0] = tokens[0].lstrip()
    y, denom = get_summed_log_unigram_probability(tokens)
    # print(f"Line: {line}, Log Sentence Prob: {x}, Sum Log Unigram Prob: {y}, Num Tokens: {len(tokens)}")


    morcela = (x - UNIGRAM_COEFFECIENT_BETA * y + LENGTH_COEFFECIENT_GAMMA) / denom # Using MORCELA formula, which is SLOR with coeffecients 
    slor = (x - y) / denom

    print(f"Stats for sentence '{sent}' : morcela {morcela}, slor {slor}")

print("----")
for sent in [a, b]:
    x = get_log_sentence_probability(sent, pythia_model, pythia_tokenizer, device)
    tokens = pythia_tokenizer.tokenize(sent)

    # tokens = [token.replace("Ġ", " ") for token in tokens] # remove the Ġ character that indicates a space in GPT-2 tokenization, since the unigram scores are for the base token without the Ġ
    # tokens[0] = tokens[0].lstrip()
    y, denom = get_summed_log_unigram_probability(tokens)
    # print(f"Line: {line}, Log Sentence Prob: {x}, Sum Log Unigram Prob: {y}, Num Tokens: {len(tokens)}")


    morcela = (x - UNIGRAM_COEFFECIENT_BETA * y + LENGTH_COEFFECIENT_GAMMA) / denom # Using MORCELA formula, which is SLOR with coeffecients 
    slor = (x - y) / denom
    print(f"Stats without replacing Ġ for sentence '{sent}' : morcela {morcela}, slor {slor}")


sys.exit()


test_sentence = "He is a boy"
test_sentence_tokens = tokenizer.encode(test_sentence)


print(tokenizer.tokenize(test_sentence))
sys.exit()
labels = test_sentence_tokens[1:]

input_ids = tokenizer.encode(test_sentence_tokens, return_tensors='pt').to(device)
output = model(input_ids, labels = input_ids)

print(input_ids.shape)
print(output.logits.shape)
sys.exit()
log_prob = -output.loss

print(log_prob)

print(math.exp(log_prob))
print(math.pow(10, log_prob))


sys.exit()
next_token_logits = output.logits


probabilities = F.softmax(next_token_logits, dim=2)

print(probabilities)

print(torch.sum(probabilities))
