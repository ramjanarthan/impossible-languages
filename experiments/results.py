import os
import csv
from datetime import datetime
from typing import List

from data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME, IMPOSSIBLE_MODEL_CHECKPOINTS

RESULTS_DIR = 'experiments/output/v2'
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results.csv')
TRAJECTORY_DIR = os.path.join(RESULTS_DIR, 'trajectory')

CSV_COLUMNS = [
    'model name',
    'checkpoint',
    'grammatical phenomenon',
    'dataset language',
    'accuracy',
    'perplexity good',
    'perplexity bad',
    'dataset path',
    'timestamp',
]

MODEL_AND_LANGUAGE_OPTIONS = list(PERTURBATION_TO_HF_MODEL_NAME.keys())
DEFAULT_MODEL_CHECKPOINT = IMPOSSIBLE_MODEL_CHECKPOINTS[-1]

def ensure_results_csv_exists(results_csv: str):
    os.makedirs(TRAJECTORY_DIR, exist_ok=True)
    if not os.path.exists(results_csv):
        with open(results_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_result(
    results_csv: str,
    model_name: str,
    checkpoint: str,
    grammatical_phenomenon: str,
    dataset_language: str,
    accuracy: float,
    perplexity_good: float,
    perplexity_bad: float,
    dataset_path: str,
):
    """
    Appends a result to the results.csv file. Timestamp is set to current local time.
    """
    ensure_results_csv_exists(results_csv)
    timestamp = datetime.now().isoformat()
    row = {
        'model name': model_name,
        'checkpoint': checkpoint,
        'grammatical phenomenon': grammatical_phenomenon,
        'dataset language': dataset_language,
        'accuracy': accuracy,
        'perplexity good': perplexity_good,
        'perplexity bad': perplexity_bad,
        'dataset path': dataset_path,
        'timestamp': timestamp,
    }
    with open(results_csv, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

def get_valid_model_and_language_options() -> List[str]:
    """
    Returns the list of valid model names and dataset languages.
    """
    return MODEL_AND_LANGUAGE_OPTIONS
