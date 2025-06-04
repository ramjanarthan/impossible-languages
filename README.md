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
conda install transformers nltk jsonlines
pip install torch
```

## Generating data:

The data generation scripts are in the `data_generation/generation_projects/impossible_blimp` directory. The output will be in the `data_generation/outputs/impossible_blimp` directory.

To generate data, run:

```bash
python -m data_generation.generation_projects.impossible_blimp.<generator name>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.english_to_local_shuffle_three_for_anaphor_agreement_gender
```

## Running experiments:

The experiments are in the `experiments` directory. The output will be in the `experiments/output` directory.

To run an experiment, run:

```bash
python -m experiments.<experiment name>
```

For example:
```bash
python -m experiments.local_shuffle_three_to_english_for_anaphor_agreement_numerical_experiment
```