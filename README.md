# When transformers learn “impossible” languages, what do they learn?

This repository contains all code used as part of the paper ("When transformers learn “impossible” languages, what do they learn?") [TODO: Insert link]

If you use our code, please cite our paper:
[TODO: Insert citation]

This repository contains all code and data necessary to fully replicate the results of Experiments 1 and 2 in our paper. This README introduces them as follows:
1. Setup and useful information
2. Generating impossible BLiMP datasets
3. Evaluating impossible models on these datasets
4. Generating outputs from impossible models and 'inverting' them
5. Evaluating impossible model generations
6. Visualisation of results

2 and 3 correspond to Experiment 1, while 4 and 5 cover Experiment 2.

## Setup and useful information:
1. Clone the repository
2. Setup conda environment:
```bash
conda create -n impossible-languages python=3.13.2
conda activate impossible-languages
```

3. Install dependencies:
```bash
conda install transformers nltk jsonlines matplotlib seaborn
pip install torch spacy cairosvg
python -m spacy download en_core_web_sm
```

For this project, we focus on the following impossible language options:
- english ("shuffle-control" in the original Kallini et al. paper)
- shuffle_nondeterministic
- shuffle_deterministic21
- shuffle_local3
- shuffle_local5
- shuffle_local10
- shuffle_even_odd
- reverse_control
- reverse_partial
- reverse_full

Consequently, these are the valid options permitted whenever the argument `impossible_language_option` is mentioned in this README. The mapping from an impossible language option to its perturbation function can be found in `data_generation/utils/impossible_utils.py`.

## Generating impossible BLiMP dataset

The impossible BLiMP datasets are modified versions of the base BLiMP dataset created by applying a perturbation function to each sentence of the minimal pair in base dataset. The script `data_generation/generation_projects/impossible_blimp/modify_dataset.py` is a handy utilty to do this, taking as arguments the base dataset path, an ```impossible_language_option```, and an output path to store the generated dataset. All versions of the original BLiMP datasets are copied and available in `data_generation/outputs/blimp/`. 

To generate the impossible dataset, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.modify_dataset <path/to/base_dataset.jsonl> <impossible_language_option>
```

For example:

```bash
python -m data_generation.generation_projects.impossible_blimp.modify_dataset data_generation/outputs/impossible_blimp/v3/distractor_agreement_relative_clause_20250712_172752%filtered.jsonl shuffle_nondeterministic
```
### Ensuring token length parity
We discard minimal pairs that will be tokenized to equal lengths by the impossible language tokenizers from the impossible BLiMP dataset prior to evaluation, since such pairs could become less minimally distinct in the impossible languages due to rules which depend on linear position/number of tokens. 

The script `data_generation/generation_projects/impossible_blimp/filter_dataset.py` is a handy utilty to do this, taking as arguments the filepath of a minimal pair dataset and an output path. It filters out pairs that do not yield equal token lengths, and writes the filtered dataset to a new file with the suffix `filtered`. This will be an invariant assumed by other scripts during evaluation to prevent accidentally including uneven minimal pairs.

To filter an impossible dataset, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.filter_dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.filter_dataset data_generation/outputs/impossible_blimp/v3/distractor_agreement_rc_20250616_174118.jsonl
```

NOTE: All impossible datasets based on BLiMP datasets have already been generated and filtered. They are located in `data_generation/outputs/impossible_blimp/v3`.


## Evaluating impossible models on these datasets

An experiment is defined as measuring the accuracy of a model when applied to a BLiMP style dataset. We computed accuracy as the percentage of minimal pairs
where the log likelihood of the grammatical sentence was larger than the ungrammatical sentence. The script ```experiments/experiment.py``` is a handy utility to do this.

To run an experiment, run the following command in the root project directory:

```bash
python -m experiments.experiment --results_csv <path/to/results.csv> --model_name <impossible_language_option> --dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m experiments.experiment --results_csv experiments/output/results.csv --model_name shuffle_nondeterministic --dataset data_generation/outputs/impossible_blimp/v3/anaphor_number_agreement_20250617_153306%shuffle_deterministic21.jsonl
```

NOTE: All experiments have been run on impossible datasets, and results are located in `experiments/output/results.csv`.

## Generating outputs from impossible models and 'inverting' them
To evaluate impossible models' generative capacity, leverage the fact that most of impossible languages can be deterministically reverted to English. We first generated 1000 sentences of up to 50 tokens from each model using a multinomial sampling strategy over their vocabularies, stored in `data_generation/outputs/impossible_generations/raw`. We then undo the perturbations and store the decoded generations to be evaluated in `data_generation/outputs/impossible_generations/corrected`.

The script `data_generation/generation_projects/impossible_generations/generate.py` is a handy utility to do this. 

The defintions of the functions used to undo the perturbations can be found `data_generation/utils/impossible_utils.py` in ```UNDO_PERTURBATIONS```.

NOTE: All generation results are located in `TBD`. 
TODO: Insert all generations used

##  Evaluating impossible model generations

To evaluate a generation's acceptability, we computed the perplexity per token using a pretrained LLM (GPT2 Large).
The script `evaluation/fluency/evaluate_fluency.py` is a handy utility to do this. 

NOTE: All evaluation results are located in `evaluation/fluency/fluency_scores_gpt2.csv`. 

## Visualisation of results

To make viewing experiment results easier, a GUI is provided in the `gui` directory. This allows comparing results between models across specific/all BLiMP grammatical phenomena, or comparing results across all models on different BLiMP grammatical phenomena.

To run the GUI, run the following command in the 'gui' project directory:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The GUI will be available at: http://localhost:8000
The GUI loads data from the results file:  (`experiments/output/results.csv`).