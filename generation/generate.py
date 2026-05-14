from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import torch
from tqdm import tqdm
from utils.impossible_utils import (
    PERTURBATION_TO_HF_MODEL_NAME,
    VALID_UNDO_PERTURBATION_KEYS,
    UNDO_PERTURBATIONS,
    PERTURBATIONS,
)


def preprocess_token_tensor(tensor, eos_token_id) -> list:
    output = tensor[0].tolist()[1:]  # Remove the BOS token

    if output[-1] == eos_token_id:
        output = output[:-1]  # Remove the EOS token if it's there
    return output


def generate_samples(model_name, num_lines, output_files):
    assert len(output_files) == 2  # one for raw (first), one for corrected (second)
    raw_output_path, corrected_output_path = output_files

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if model_name in VALID_UNDO_PERTURBATION_KEYS:
        postprocess_fn = UNDO_PERTURBATIONS[model_name]["perturbation_function"]
    else:
        postprocess_fn = None

    raw, corrected = generate_samples_inner(
        model_name=model_name,
        num_lines=num_lines,
        device=device,
        postprocess_fn=postprocess_fn,
    )
    # print lines to file
    with (
        open(raw_output_path, "w") as raw_file,
        open(corrected_output_path, "w") as corrected_file,
    ):
        for raw_line, corrected_line in zip(raw, corrected):
            raw_file.write(raw_line + "\n")
            corrected_file.write(corrected_line + "\n")


def generate_samples_inner(
    model_name,
    num_lines,
    device,
    postprocess_fn=None,
):
    model_id = PERTURBATION_TO_HF_MODEL_NAME[model_name]

    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    model.eval()

    input_ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    outputs_raw = []
    outputs_corrected = []

    for i in tqdm(range(num_lines), desc=f"Generating for {model_id}"):
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        tokens = output[0]
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        corrected_text = text
        if postprocess_fn is not None:
            try:
                tokens = preprocess_token_tensor(output, tokenizer.eos_token_id)
                corrected = postprocess_fn(tokens)
                corrected_text = tokenizer.decode(corrected, skip_special_tokens=True)
            except Exception as e:
                print(f"Error applying undo perturbation: {text}")

        outputs_raw.append(text)
        outputs_corrected.append(corrected_text)
    return outputs_raw, outputs_corrected
