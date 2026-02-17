from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
import pickle
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_UNDO_PERTURBATION_KEYS, UNDO_PERTURBATIONS, PERTURBATIONS

NUM_LINES = 1000
RAW_OUTPUT_DIR = "data_generation/outputs/impossible_generations/raw/"
CORRECTED_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"

for perturbation in VALID_UNDO_PERTURBATION_KEYS:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = PERTURBATIONS[perturbation]['gpt2_tokenizer']   

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]])
    attention_mask = torch.ones_like(input_ids)

    raw_outputs = []
    corrected_outputs = []

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {perturbation}"):
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_outputs.append(generated_text)

        undo_input = output[0].tolist()[1:] # Remove the BOS token

        if undo_input[-1] == tokenizer.eos_token_id:
            undo_input = undo_input[:-1] # Remove the EOS token if it's there
        try:
            corrected_output = UNDO_PERTURBATIONS[perturbation]['perturbation_function'](undo_input)
        except Exception as e:
            print(f"Error applying undo perturbation for {perturbation} on generated output: {generated_text}")
            print(undo_input, e)
        corrected_text = UNDO_PERTURBATIONS[perturbation]['gpt2_tokenizer'].decode(corrected_output, skip_special_tokens=True)
        corrected_outputs.append(corrected_text)

    # Save generated text to file
    with open(RAW_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        f.write("\n".join(raw_outputs))
    
    with open(CORRECTED_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        f.write("\n".join(corrected_outputs))
    
    print(f"Generated {NUM_LINES} sentences for {perturbation}.")

for perturbation in ['english', 'shuffle_nondeterministic']:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id)
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)   

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]])

    raw_outputs = []
    corrected_outputs = []

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {perturbation}"):
        output = model.generate(input_ids, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_outputs.append(output[0])

    # Save generated text to file
    # pickle the raw tokens
    with open(RAW_OUTPUT_DIR + f"{perturbation}.pkl", "wb") as f:
        pickle.dump(raw_outputs, f)

    # with open(CORRECTED_OUTPUT_DIR + f"{perturbation}.p", "w+") as f:
    #     # f.write("\n".join(raw_outputs))
    #     # save 

    
    print(f"Generated {NUM_LINES} sentences for {perturbation}.")