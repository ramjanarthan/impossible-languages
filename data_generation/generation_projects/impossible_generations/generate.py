from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, VALID_UNDO_PERTURBATION_KEYS, UNDO_PERTURBATIONS, PERTURBATIONS
import json

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

def write_dict_to_file(dict, filepath):
    with open(filepath, "w+") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)

for perturbation in VALID_UNDO_PERTURBATION_KEYS:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = PERTURBATIONS[perturbation]['gpt2_tokenizer']   

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    raw_outputs = {}
    corrected_outputs = {}

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {perturbation}"):
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        undo_input = preprocess_token_tensor(output, tokenizer.eos_token_id)

        try:
            corrected_output = UNDO_PERTURBATIONS[perturbation]['perturbation_function'](undo_input)
        except Exception as e:
            print(f"Error applying undo perturbation for {perturbation} on generated output: {generated_text}")
            print(undo_input, e)
        
        corrected_text = UNDO_PERTURBATIONS[perturbation]['gpt2_tokenizer'].decode(corrected_output, skip_special_tokens=True)

        raw_outputs[i] = generated_text
        corrected_outputs[i] = corrected_text

    # Save generated text to file
    write_dict_to_file(raw_outputs, RAW_OUTPUT_DIR + f"{perturbation}.txt")
    write_dict_to_file(corrected_outputs, CORRECTED_OUTPUT_DIR + f"{perturbation}.txt")

    print(f"Generated {NUM_LINES} sentences for {perturbation}.")

for perturbation in ['english', 'shuffle_nondeterministic']:
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    raw_outputs = {}

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {perturbation}"):
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_outputs[i] = generated_text

    # Save generated text to file
    write_dict_to_file(raw_outputs, CORRECTED_OUTPUT_DIR + f"{perturbation}.txt")

    print(f"Generated {NUM_LINES} sentences for {perturbation}.")

external_models = ["openai-community/gpt2"]
for model_id in external_models:
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)

    # Start with just the BOS token (no initial prompt)
    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    raw_outputs = {}

    for i in tqdm(range(NUM_LINES), desc=f"Generating for {model_id}"):
        output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        raw_outputs[i] = generated_text

    model_name = model_id.replace("/", "_")
    
    # Save generated text to file
    write_dict_to_file(raw_outputs, CORRECTED_OUTPUT_DIR + f"{model_name}.txt")
    
    print(f"Generated {NUM_LINES} sentences for {model_name}.")