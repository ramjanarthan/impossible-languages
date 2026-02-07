from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from tqdm import tqdm
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_UNDO_PERTURBATION_KEYS, UNDO_PERTURBATIONS, PERTURBATIONS

NUM_LINES = 50
RAW_OUTPUT_DIR = "data_generation/outputs/impossible_generations/raw/"
CORRECTED_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def preprocess_token_tensor(tensor, eos_token_id) -> list:
    output = tensor[0].tolist()[1:] # Remove the BOS token

    if output[-1] == eos_token_id:
        output = output[:-1] # Remove the EOS token if it's there
    return output

for perturbation in VALID_UNDO_PERTURBATION_KEYS:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = PERTURBATIONS[perturbation]['gpt2_tokenizer']   

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    raw_outputs = []
    corrected_outputs = []

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {perturbation}"):
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # raw_outputs.append(output[0])

        undo_input = preprocess_token_tensor(output, tokenizer.eos_token_id)

        try:
            corrected_output = UNDO_PERTURBATIONS[perturbation]['perturbation_function'](undo_input)
        except Exception as e:
            print(f"Error applying undo perturbation for {perturbation} on generated output: {generated_text}")
            print(undo_input, e)
        
        corrected_text = UNDO_PERTURBATIONS[perturbation]['gpt2_tokenizer'].decode(corrected_output, skip_special_tokens=True)

        raw_outputs.append(undo_input)
        corrected_outputs.append(corrected_output)
        # corrected_outputs.append(corrected_text)

    # Save generated text to file
    with open(RAW_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        for line in raw_outputs:
            f.write(f"{line}\n")
    
    with open(CORRECTED_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        for line in corrected_outputs:
            f.write(f"{line}\n")
    
    print(f"Generated {NUM_LINES} sentences for {perturbation}.")

for perturbation in ['english', 'shuffle_nondeterministic']:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    raw_outputs = []

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {perturbation}"):
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_outputs.append(preprocess_token_tensor(output, tokenizer.eos_token_id))

    # Save generated text to file
    with open(CORRECTED_OUTPUT_DIR + f"{perturbation}.txt", "w+") as f:
        for line in raw_outputs:
            f.write(f"{line}\n")
    
    print(f"Generated {NUM_LINES} sentences for {perturbation}.")