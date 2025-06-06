from data_generation.utils.impossible_utils import perturb_shuffle_local, gpt2_original_tokenizer
from transformers import GPT2Tokenizer

file  = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/small_test/special.test"

with open(file, "r") as f:
    lines = f.readlines()

output_tokens = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages/compare_original/special_output_tokens.test"
output_decode = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages/compare_original/special_output_decode.test"

# for line in lines:
#     perturbed = perturb_shuffle_local(line, seed=0, window=3)
#     # write to file
#     with open(output_tokens, "w") as f:
#         for token in perturbed:
#             f.write(f"{token} ")
#         f.write("\n")

#     with open(output_decode, "w") as f:
#         decoded = gpt2_original_tokenizer.decode(perturbed, skip_special_tokens=True)
#         f.write(decoded + "\n")

with open(output_decode, "r") as f:
    decode_lines = f.readlines()

model_name = "mission-impossible-lms/local-shuffle-w3-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

for line in decode_lines:
    encoded = tokenizer(line)
    print(f"encoded: {encoded}")
    
    encoded_gpt2 = gpt2_original_tokenizer(line)
    print(f"encoded_gpt2: {encoded_gpt2}")

    print(f"comparison: {encoded == encoded_gpt2}")