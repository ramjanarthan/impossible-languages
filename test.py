from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import sys
import math
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_UNDO_PERTURBATION_KEYS, UNDO_PERTURBATIONS, PERTURBATIONS
import json


model_id = "EleutherAI/pythia-14M"
device = "mps"
# model, tokenizer = GPT2LMHeadModel.from_pretrained(model_id).to(device), GPT2Tokenizer.from_pretrained(model_id)

# Setup GPT-2 small pre-trained model
model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-14m").to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-14m")
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
print(tokenizer.tokenize(a))

print(b)
print(tokenizer.tokenize(b))

# print("----")
# for str in ["say", "Ġsay", " say"]:
#     print(f"Logprob for '{str}' :", in_mem_unigram_scores[str])

print("-----")

for sent in [a, b]:
    x = get_log_sentence_probability(sent, model, tokenizer, device)
    tokens = tokenizer.tokenize(sent)

    tokens = [token.replace("Ġ", " ") for token in tokens] # remove the Ġ character that indicates a space in GPT-2 tokenization, since the unigram scores are for the base token without the Ġ
    tokens[0] = tokens[0].lstrip()
    y, denom = get_summed_log_unigram_probability(tokens)
    # print(f"Line: {line}, Log Sentence Prob: {x}, Sum Log Unigram Prob: {y}, Num Tokens: {len(tokens)}")


    morcela = (x - UNIGRAM_COEFFECIENT_BETA * y + LENGTH_COEFFECIENT_GAMMA) / denom # Using MORCELA formula, which is SLOR with coeffecients 
    slor = (x - y) / denom

    print(f"Stats for sentence '{sent}' : morcela {morcela}, slor {slor}")

print("----")
for sent in [a, b]:
    x = get_log_sentence_probability(sent, model, tokenizer, device)
    tokens = tokenizer.tokenize(sent)

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
