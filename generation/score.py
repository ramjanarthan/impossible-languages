from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.impossible_utils import get_device


def get_log_sentence_probability(line, model, tokenizer, device) -> float:
    encoded_line = tokenizer(line, return_tensors="pt", return_attention_mask=True).to(
        device
    )

    output = model(
        encoded_line.input_ids,
        labels=encoded_line.input_ids,
    )

    loss_ce = output.loss

    log_sentence_probability = -loss_ce
    return log_sentence_probability.item()


def score_generation_file(
    model_name,
    input_path,
    output_path,
):
    device = get_device()
    model_id = "gpt2" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()

    # print("Computing scores for file : ", file)
    # Read the content of the file line by line
    with open(input_path, "r") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    data = []
    for line in tqdm(lines, leave=False, desc=f"Scoring {input_path}"):

        # remove any 🅁 and track if we did:
        hadR = False
        if "🅁" in line:
            line_cleaned = line.replace("🅁", "")
            hadR = True
        else:
            line_cleaned = line

        x = get_log_sentence_probability(line_cleaned, model, tokenizer, device)

        tokens = tokenizer.tokenize(line_cleaned)
        data.append(
            {
                "perturbation": model_name,
                "generation": line,
                "generation_cleaned": line_cleaned,
                "hadR": hadR,
                "perplexity": np.exp(-x / len(tokens)),
                "ntokens": len(tokens),
            }
        )
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
