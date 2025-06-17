import os
import csv
from datetime import datetime
from typing import List

from impossible_languages.data_generation.utils.impossible_utils import PERTURBATION_TO_HF_MODEL_NAME

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'v2')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results.csv')

CSV_COLUMNS = [
    'model name',
    'grammatical phenomenon',
    'dataset language',
    'accuracy',
    'perplexity good',
    'perplexity bad',
    'dataset path',
    'timestamp',
]

MODEL_AND_LANGUAGE_OPTIONS = list(PERTURBATION_TO_HF_MODEL_NAME.keys())

def ensure_results_csv_exists():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()

def append_result(
    model_name: str,
    grammatical_phenomenon: str,
    dataset_language: str,
    accuracy: float,
    perplexity_good: float,
    perplexity_bad: float,
    dataset_path: str,
    timestamp: str = None,
):
    """
    Appends a result to the results.csv file. Timestamp is set to current local time if not provided.
    """
    ensure_results_csv_exists()
    if timestamp is None:
        # Use the provided current local time (2025-06-17T11:07:56+01:00)
        timestamp = datetime.now().isoformat()
    row = {
        'model name': model_name,
        'grammatical phenomenon': grammatical_phenomenon,
        'dataset language': dataset_language,
        'accuracy': accuracy,
        'perplexity good': perplexity_good,
        'perplexity bad': perplexity_bad,
        'dataset path': dataset_path,
        'timestamp': timestamp,
    }
    with open(RESULTS_CSV, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)

def get_valid_model_and_language_options() -> List[str]:
    """
    Returns the list of valid model names and dataset languages.
    """
    return MODEL_AND_LANGUAGE_OPTIONS
