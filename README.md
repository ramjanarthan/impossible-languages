# impossible-languages

## Setup:
1. Clone the repository
2. Setup conda environment:
```bash
conda create -n impossible-languages python=3.13.2
conda activate impossible-languages
```

3. Install dependencies:
```bash
conda install transformers
pip install torch
```

4. Run the evaluation script:
```bash
python experiments/<experiment_file>.py
```