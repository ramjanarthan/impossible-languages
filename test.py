from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn.functional as F

model_id = "mission-impossible-lms/no-shuffle-gpt2"
device = "mps"
model, tokenizer = GPT2LMHeadModel.from_pretrained(model_id).to(device), GPT2Tokenizer.from_pretrained(model_id)

empty_prompt = "Hi " #tokenizer.bos_token

test_sentence = "He is a boy"
test_sentence_tokens = tokenizer.encode(test_sentence)

input_ids = tokenizer.encode(empty_prompt, return_tensors='pt').to(device)
output = model(input_ids)

next_token_logits = output.logits[:, -1, :]
probabilities = F.log_softmax(next_token_logits, dim=0)

print(torch.sum(probabilities))