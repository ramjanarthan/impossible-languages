from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "index.html")
import re
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import glob

app = FastAPI(title="Linguistic Experiments Viewer")

@dataclass
class ExperimentResult:
    """Data class to hold experiment results"""
    dataset: str
    model1: str
    model2: str
    accuracy1: float
    accuracy2: float
    timestamp: str
    filename: str
    language1: str
    language2: str
    phenomenon: str
    perplexities: Dict[str, float]
    comparison_counts: Dict[str, int]
    total_pairs: int

class ExperimentParser:
    """Parser for experiment result files"""
    
    def __init__(self, experiments_path: str = "../experiments/output/v1"):
        self.experiments_path = experiments_path
    
    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Parse filename to extract experiment parameters"""
        # Pattern: {lang1}_to_{lang2}_for_{phenomenon}_experiment_eval_{timestamp}.txt
        pattern = r'(.+)_to_(.+)_for_(.+)_experiment_eval_(\d{8}_\d{6})\.txt'
        match = re.match(pattern, filename)
        
        if not match:
            return None
            
        return {
            'language1': match.group(1),
            'language2': match.group(2), 
            'phenomenon': match.group(3),
            'timestamp': match.group(4)
        }
    
    def parse_file_content(self, filepath: str) -> Optional[Dict]:
        """Parse the content of an experiment file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract dataset
            dataset_match = re.search(r'Dataset: (.+)', content)
            if not dataset_match:
                return None
            dataset_full_path = dataset_match.group(1).strip()
            dataset_filename = os.path.basename(dataset_full_path)
            dataset_brief_name = os.path.splitext(dataset_filename)[0]
            dataset = dataset_brief_name  # Use the brief name
            
            # Extract models
            model1_match = re.search(r'Model 1: (.+)', content)
            model2_match = re.search(r'Model 2: (.+)', content)
            if not model1_match or not model2_match:
                return None
            model1 = model1_match.group(1).strip()
            model2 = model2_match.group(1).strip()
            
            # Extract total pairs
            pairs_match = re.search(r'Total Parallel Pairs Processed: (\d+)', content)
            if not pairs_match:
                return None
            total_pairs = int(pairs_match.group(1))
            
            # Extract accuracies
            acc1_match = re.search(r'Accuracy for Model 1 \(.+?\) on Dataset A: ([\d.]+)%', content)
            acc2_match = re.search(r'Accuracy for Model 2 \(.+?\) on Dataset B: ([\d.]+)%', content)
            if not acc1_match or not acc2_match:
                return None
            accuracy1 = float(acc1_match.group(1))
            accuracy2 = float(acc2_match.group(1))
            
            # Extract comparison counts
            comparison_counts = {}
            both_correct_match = re.search(r'Both Models Correct: (\d+)', content)
            model1_only_match = re.search(r'mission-impossible-lms/\S+ Only Correct: (\d+)', content)
            model2_only_match = re.search(r'mission-impossible-lms/\S+ Only Correct: (\d+)', content)
            neither_match = re.search(r'Neither Model Correct: (\d+)', content)
            
            if both_correct_match:
                comparison_counts['both_correct'] = int(both_correct_match.group(1))
            if neither_match:
                comparison_counts['neither_correct'] = int(neither_match.group(1))
            
            # Extract perplexities
            perplexities = {}
            perp_matches = re.findall(r'([AB]_(?:good|bad)_m[12]): ([\d.]+)', content)
            for key, value in perp_matches:
                perplexities[key] = float(value)
            
            return {
                'dataset': dataset,
                'model1': model1,
                'model2': model2,
                'accuracy1': accuracy1,
                'accuracy2': accuracy2,
                'total_pairs': total_pairs,
                'comparison_counts': comparison_counts,
                'perplexities': perplexities
            }
            
        except Exception as e:
            print(f"Error parsing file {filepath}: {e}")
            return None
    
    def load_experiments(self) -> Dict[str, List[ExperimentResult]]:
        """Load and parse all experiment files"""
        experiments = {}
        
        if not os.path.exists(self.experiments_path):
            print(f"Experiments directory {self.experiments_path} does not exist.")
            return experiments
        
        files = glob.glob(os.path.join(self.experiments_path, "*.txt"))
        parsed_experiments = {}
        
        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Parse filename
            filename_data = self.parse_filename(filename)
            if not filename_data:
                continue
            
            # Parse file content
            content_data = self.parse_file_content(filepath)
            if not content_data:
                continue
            
            # Create experiment result
            experiment = ExperimentResult(
                dataset=content_data['dataset'],
                model1=content_data['model1'],
                model2=content_data['model2'],
                accuracy1=content_data['accuracy1'],
                accuracy2=content_data['accuracy2'],
                timestamp=filename_data['timestamp'],
                filename=filename,
                language1=filename_data['language1'],
                language2=filename_data['language2'],
                phenomenon=filename_data['phenomenon'],
                perplexities=content_data['perplexities'],
                comparison_counts=content_data['comparison_counts'],
                total_pairs=content_data['total_pairs']
            )
            
            # Use prefix as key for deduplication
            prefix = f"{filename_data['language1']}_to_{filename_data['language2']}_for_{filename_data['phenomenon']}"
            
            # Keep only the latest timestamp for each prefix
            if prefix not in parsed_experiments or filename_data['timestamp'] > parsed_experiments[prefix].timestamp:
                parsed_experiments[prefix] = experiment
        
        # Convert parsed_experiments dictionary to a list for initial sorting
        initial_sorted_list = list(parsed_experiments.values())

        # Sort experiments: by phenomenon, then by language pair (normalized), then by language1
        initial_sorted_list.sort(key=lambda exp: (
            exp.phenomenon,
            tuple(sorted((exp.language1, exp.language2))),
            exp.language1 # Ensures exp1 (e.g., A->B) comes before exp2 (e.g., B->A) if A < B
        ))

        final_grouped_experiments = {}
        i = 0
        while i < len(initial_sorted_list):
            current_exp = initial_sorted_list[i]
            phenomenon = current_exp.phenomenon

            if phenomenon not in final_grouped_experiments:
                final_grouped_experiments[phenomenon] = []

            # Check for a mirror with the next experiment
            if i + 1 < len(initial_sorted_list):
                next_exp = initial_sorted_list[i+1]
                is_mirror = (
                    current_exp.phenomenon == next_exp.phenomenon and
                    current_exp.dataset == next_exp.dataset and # Compare brief dataset names
                    current_exp.language1 == next_exp.language2 and
                    current_exp.language2 == next_exp.language1
                )

                if is_mirror:
                    # Ensure current_exp.language1 is lexicographically smaller for consistent ordering in combined obj
                    exp1_data = asdict(current_exp) # lang1 -> lang2
                    exp2_data = asdict(next_exp)   # lang2 -> lang1
                    
                    combined_entry = {
                        "type": "combined",
                        "phenomenon": phenomenon,
                        "dataset": current_exp.dataset,
                        "language_pair_sorted": tuple(sorted((current_exp.language1, current_exp.language2))),
                        "experiment_A_to_B": exp1_data,
                        "experiment_B_to_A": exp2_data
                    }
                    final_grouped_experiments[phenomenon].append(combined_entry)
                    i += 2  # Move past both current and next experiment
                    continue
            
            # Not a mirror or it's the last experiment, add as single
            single_entry = {
                "type": "single",
                "experiment": asdict(current_exp)
            }
            final_grouped_experiments[phenomenon].append(single_entry)
            i += 1
            
        return final_grouped_experiments

# Initialize parser
parser = ExperimentParser()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse(INDEX_HTML_PATH)

@app.get("/api/experiments/v1")
async def get_experiments_v1():
    """API endpoint to get experiment data"""
    # parser.load_experiments() now returns the final structure ready for JSON serialization
    experiments_data = parser.load_experiments()
    return experiments_data

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)