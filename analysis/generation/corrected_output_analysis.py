from tqdm import tqdm
from transformers import AutoTokenizer
from data_generation.utils.impossible_utils import VALID_PERTURBATION_KEYS, PERTURBATION_TO_HF_MODEL_NAME
import numpy as np

CORRECTED_OUTPUT_DIR = "data_generation/outputs/impossible_generations/corrected/"

for perturbation in VALID_PERTURBATION_KEYS:
    file_path = CORRECTED_OUTPUT_DIR + perturbation + ".txt"
    model_id = PERTURBATION_TO_HF_MODEL_NAME[perturbation]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    lengths = []
    with open(file_path, "r") as file:
        lines = [line for line in file]
        for line in tqdm(lines, desc=f"Analysing {perturbation}"):
            length = len(tokenizer.tokenize(line))
            lengths.append(length)

    print(f"Mean length & std for {perturbation} generation: {np.mean(lengths)}, {np.std(lengths)}")
