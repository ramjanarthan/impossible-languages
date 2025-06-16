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

The generation is split into two phases: generating a base dataset, and generating the impossible version of the dataset. For the first part, I adopt the BLiMP generation method, and for the second part, I apply a perturbation function to the base dataset.

### Generating base dataset

To generate the base dataset, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.v2.<generator name>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.v2.distractor_agreement_rc
```

### Generating impossible dataset

The impossible datasets are modified versions of the base dataset, where the perturbation function is applied to the base dataset.

To generate the impossible dataset, run the following command in the root project directory:


## Running experiments:

The experiments are in the `experiments` directory. The output will be in the `experiments/output` directory.

To run an experiment, run the following command in the root project directory:

```bash
python -m experiments.v2.<experiment name>
```

For example:
```bash
python -m experiments.v2.distractor_agreement_rc
```

## Naming convention

Where applicable, the file and class names will follow this convention:

```
{First Language Name}_to_{Second Language Name}_for_{Grammatical Phenomenon}
```

The order is significant because the dataset is split into two parts, and the models are evaluated on the parts in the same order.

For example:

```
english_to_local_shuffle_three_for_anaphor_agreement_gender_experiment_eval_20250606_173847
```

refers to an experiment evaluation file comparing an English Model to a Local Shuffle Three Model for Anaphor Agreement Gender (in that order).

## GUI

To make viewing experiment results easier, a GUI is provided in the `gui` directory.

To run the GUI, run the following command in the 'gui' project directory:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The GUI will be available at: http://localhost:8000