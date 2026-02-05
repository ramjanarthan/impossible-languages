from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F
import sys
import math

model_id = "mission-impossible-lms/no-shuffle-gpt2"
device = "mps"
model, tokenizer = GPT2LMHeadModel.from_pretrained(model_id).to(device), GPT2Tokenizer.from_pretrained(model_id)

empty_prompt = tokenizer.bos_token

test_sentence = "He is a boy"
test_sentence_tokens = tokenizer.encode(test_sentence)
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
