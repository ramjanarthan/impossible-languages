from data_generation.utils.impossible_utils import perturb_shuffle_local, gpt2_original_tokenizer


file  = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/external_code/mission-impossible-language-models/test/small_test/special.test"

with open(file, "r") as f:
    lines = f.readlines()

output_tokens = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages/test/special_output_tokens.test"
output_decode = "/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages/test/special_output_decode.test"

for line in lines:
    perturbed = perturb_shuffle_local(line, seed=0, window=3)
    # write to file
    with open(output_tokens, "w") as f:
        for token in perturbed:
            f.write(f"{token} ")
        f.write("\n")

    with open(output_decode, "w") as f:
        decoded = gpt2_original_tokenizer.decode(perturbed, skip_special_tokens=True)
        f.write(decoded + "\n")