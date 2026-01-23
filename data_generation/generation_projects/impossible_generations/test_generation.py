from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_PERTURBATION_KEYS

NUM_LINES = 50
OUTPUT_DIR = "data_generation/outputs/impossible_generations/"

for perturbation in VALID_PERTURBATION_KEYS:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]])

    outputs = []

    for i in range(NUM_LINES):
        output = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(generated_text)

    # Save generated text to file
    with open(OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        f.write("\n".join(outputs))

    print(f"Generated {NUM_LINES} sentences for {perturbation}.")