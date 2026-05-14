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

## 1. Setup and useful information:
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

These are the valid options permitted whenever the argument `impossible_language_option` is mentioned in the rest of this README. The mapping from an impossible language option to its perturbation function can be found in `data_generation/utils/impossible_utils.py`.

### 1.1 Dataset naming convention and enforcement
Throughout this project, the dataset filenames have a prefix of ```<task_name>_<timestamp>``` for readability and consistency. 

When evaluating a dataset, the filenames should be of the format ```<task_name>_<timestamp>_filtered_<language?>.jsonl``` (language is unspecified when English). The logic to enforce this can be found in the ```_parse_dataset_filename``` method in `evaluation/evaluate.py`.

## 2. Generating impossible BLiMP datasets

The impossible BLiMP datasets are modified versions of the base BLiMP dataset created by applying a perturbation function to each sentence of the minimal pair in base dataset. All versions of the original BLiMP datasets are copied from the BLiMP code (found at [this link](https://github.com/alexwarstadt/blimp)) and available in `data_generation/outputs/blimp/`. However, prior to generating the BLiMP datasets for each impossible language, we must first ensure token length partiy for valid minimal pairs.

### 2.1 Ensuring token length parity
We discard minimal pairs that will be tokenized to unequal lengths by an impossible language tokenizer from an impossible BLiMP dataset prior to evaluation, since such pairs could become less minimally distinct in the impossible languages (due to rules which depend on linear position/number of tokens). 

The script `data_generation/generation_projects/impossible_blimp/filter_dataset.py` is a handy utilty to do this, taking as arguments the filepath of a minimal pair dataset and an output path. It filters out pairs that do not yield equal token lengths, and writes the filtered dataset to a new file with the suffix `filtered`. This naming convention will be an invariant assumed by scripts used for evaluation, to prevent accidentally evaluating on datasets that include uneven minimal pairs (in case you see the message "Experiment failed: Could not parse dataset filename:", it is likely due to this).

To filter an impossible dataset, run the following command in the project root directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.filter_dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.filter_dataset data_generation/outputs/impossible_blimp/v3/distractor_agreement_rc_20250616_174118.jsonl
```

### 2.2 Generating an impossible BLiMP dataset 
The script `data_generation/generation_projects/impossible_blimp/modify_dataset.py` is a handy utilty to apply a perturbation function to each sentence of the minimal pair in base dataset, taking as arguments the base dataset path and an ```impossible_language_option```. It automatically creates an output path by appending the ```impossible_language_option``` to the base dataset path, and saves the generated dataset at this path.
To generate the impossible dataset, run the following command in the project root directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.modify_dataset <path/to/base_dataset.jsonl> <impossible_language_option>
```

For example:

```bash
python -m data_generation.generation_projects.impossible_blimp.modify_dataset data_generation/outputs/impossible_blimp/v3/distractor_agreement_relative_clause_20250712_172752%filtered.jsonl shuffle_nondeterministic
```

NOTE: All impossible datasets based on BLiMP datasets have already been generated and filtered. They are located in `data_generation/outputs/impossible_blimp/v3`.


## 3. Evaluating impossible models on these datasets

An experiment is defined as measuring the accuracy of a model when evaluated on a BLiMP style dataset. We computed accuracy as the percentage of minimal pairs
where the model assigns a higher log likelihood to the grammatical sentence than the ungrammatical sentence. The script ```experiments/experiment.py``` is a handy utility to do this, taking as arguments the file path of a CSV file to store the results, the model (referred to by name), and the dataset to evaluate it on. 

To run an experiment, run the following command in the project root directory:

```bash
python -m experiments.experiment --results_csv <path/to/results.csv> --model_name <impossible_language_option> --dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m experiments.experiment --results_csv experiments/output/results.csv --model_name shuffle_nondeterministic --dataset data_generation/outputs/impossible_blimp/v3/adjunct_island_20260101_170829%filtered%shuffle_nondeterministic.jsonl
```

NOTE: All experiments have been run on impossible datasets, and results are located in `experiments/output/results.csv`.

## 4. Generating outputs from impossible models and 'inverting' them
To evaluate impossible models' generative capacity, we leverage the fact that most of impossible languages can be deterministically reverted to English. The defintions of the functions used to undo the perturbations can be found `data_generation/utils/impossible_utils.py` in ```UNDO_PERTURBATIONS```.

We first generate 1000 sentences of up to 50 tokens from each model using a multinomial sampling strategy over their vocabularies, stored in `data_generation/outputs/impossible_generations/raw`. We then undo the perturbations and store the decoded generations in `data_generation/outputs/impossible_generations/corrected` to be evaluated by an LLM.

The script `data_generation/generation_projects/impossible_generations/generate.py` is a handy utility to do this. To begin generation, run the following command in the project root:
```bash
python -m data_generation.generation_projects.impossible_generations.generate
```

NOTE: All generation results are located in `TBD`. 
TODO: Insert all generations used

## 5. Evaluating impossible model generations

To evaluate a generation's acceptability, we computed the perplexity per token using a pretrained LLM (GPT2 Large).
The script `evaluation/fluency/evaluate_fluency.py` is a handy utility to do this. To begin evaluation, run the following command in the project root:
```bash
python -m evaluation.fluency.evaluate_fluency
```


NOTE: All evaluation results are located in `evaluation/fluency/fluency_scores_gpt2.csv`. 

## 6. Visualisation of results

To make viewing experiment results easier, a GUI is provided in the `gui` directory. This allows comparing results between models across specific/all BLiMP grammatical phenomena, or comparing results across all models on different BLiMP grammatical phenomena like so:

![Accuracy Analysis grouped by Grammatical Phenomenon](gui.png)

To run the GUI, run the following command in the 'gui' directory:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The GUI will be available at: http://localhost:8000
The GUI loads data from the results file: (`experiments/output/results.csv`).