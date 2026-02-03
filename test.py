from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_id = "mission-impossible-lms/no-shuffle-gpt2"
device = "mps"
model, tokenizer = GPT2LMHeadModel.from_pretrained(model_id).to(device), GPT2Tokenizer.from_pretrained(model_id)

test_prompt = "He is a very "

input_ids = tokenizer.encode(test_prompt, return_tensors='pt').to(device)

output = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)

decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

print("Decoded output - ", decoded_output)