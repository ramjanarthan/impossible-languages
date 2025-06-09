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
    
    def __init__(self, experiments_path: str = "../experiments/output"):
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
            dataset = dataset_match.group(1).strip()
            
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
        
        # Group by phenomenon
        for experiment in parsed_experiments.values():
            phenomenon = experiment.phenomenon
            if phenomenon not in experiments:
                experiments[phenomenon] = []
            experiments[phenomenon].append(experiment)
        
        return experiments

# Initialize parser
parser = ExperimentParser()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse(INDEX_HTML_PATH)

@app.get("/api/experiments")
async def get_experiments():
    """API endpoint to get experiment data"""
    experiments = parser.load_experiments()
    
    # Convert to JSON-serializable format
    result = {}
    for phenomenon, exp_list in experiments.items():
        result[phenomenon] = [asdict(exp) for exp in exp_list]
    
    return result

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)