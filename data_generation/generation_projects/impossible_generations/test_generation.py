from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_PERTURBATION_KEYS, UNDO_PERTURBATIONS

NUM_LINES = 10
RAW_OUTPUT_DIR = "data_generation/outputs/impossible_generations/raw/"
CORRECTED_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"

for perturbation in [ "shuffle_deterministic21", "shuffle_local3", "shuffle_local5", "shuffle_local10"]:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]])

    raw_outputs = []
    corrected_outputs = []

    for i in range(NUM_LINES):
        output = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_outputs.append(generated_text)

        corrected_output = UNDO_PERTURBATIONS[perturbation]['perturbation_function'](generated_text)
        corrected_text = tokenizer.decode(corrected_output, skip_special_tokens=True)
        corrected_outputs.append(corrected_text)

    # Save generated text to file
    with open(RAW_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        f.write("\n".join(raw_outputs))
    
    with open(CORRECTED_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        f.write("\n".join(corrected_outputs))
    

    print(f"Generated {NUM_LINES} sentences for {perturbation}.")