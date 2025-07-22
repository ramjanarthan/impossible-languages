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
conda install transformers nltk jsonlines matplotlib seaborn
pip install torch spacy
python -m spacy download en_core_web_sm
```

## Impossible Language options

For this project, I focus on the following impossible language options:

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

The mapping from impossible language option to perturbation function is in the `data_generation/utils/impossible_utils.py` file. These are the options permitted where `impossible_language_option` is refered to subsequently.

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
python -m data_generation.generation_projects.impossible_blimp.v2.anaphor_gender_agreement_distance
```

### Ensuring token length parity
To ensure that the generated minimal pairs will be tokenized to equal lengths by the impossible language tokenizers, run the following command to filter out pairs that do not yield equal token lengths. This will write the filtered dataset to a new file with the suffix `filtered`, which will be an invariant assumed by other scripts to ensure token length parity.

```bash
python -m data_generation.generation_projects.impossible_blimp.filter_dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.filter_dataset data_generation/outputs/impossible_blimp/v2/distractor_agreement_rc_20250616_174118.jsonl
```

### Generating impossible dataset

The impossible datasets are modified versions of the base dataset, where the perturbation function is applied to the base dataset.

To generate the impossible dataset, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.modify_dataset <path/to/base_dataset.jsonl> <impossible_language_option>
```

For example:

```bash
python -m data_generation.generation_projects.impossible_blimp.modify_dataset data_generation/outputs/impossible_blimp/v2/distractor_agreement_relative_clause_20250712_172752%filtered.jsonl shuffle_nondeterministic
```

## Running experiments:

An experiment is defined as measuring the accuracy of a model when applied to a dataset.

To run an experiment, run the following command in the root project directory:

```bash
python -m experiments.experiment --results_csv <path/to/results.csv> --model_name <impossible_language_option> --dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m experiments.experiment --results_csv experiments/output/v2/results.csv --model_name shuffle_nondeterministic --dataset data_generation/outputs/impossible_blimp/v2/anaphor_number_agreement_20250617_153306%shuffle_deterministic21.jsonl
```

### Running Trajectory experiments:

An experiment to measure the trajectory of a model's performance at different model checkpoints.

The scripts to run these are located in `experiments/v2/trajectory/`. To run these, run the following command in the root project directory:

```bash
bash experiments/v2/trajectory/anaphor_gender_agreement.sh
```

The results will be logged to the csv path specified in the script. They were designed this way to be executed on a compute cluster.

## Analysis:

### Analysing dataset:

To analyse the dataset stats related to sentence length, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.analyse_dataset <path/to/dataset.jsonl>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.analyse_dataset data_generation/outputs/impossible_blimp/v2/adjunct_island_20250623_165451%filtered.jsonl
```

To analyse all datasets in one go, you can run the `data_generation/generation_projects/impossible_blimp/batch_analyse_dataset.sh` script.

Sample output:

```bash
Statistics for 'sentence_good' lengths in data_generation/outputs/impossible_blimp/v2/adjunct_island_20250623_165451%filtered.jsonl:
  Count:  1000
  Mean:   8.41
  Median: 8.00
  Min:    7
  Max:    14
```

To analyse the dataset to check if minimal pairs yield equal token lengths for the impossible language tokenizers, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.analyse_token_lengths <path/to/dataset.jsonl>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.analyse_token_lengths data_generation/outputs/impossible_blimp/v2/adjunct_island_20250623_165451.jsonl
```

This script will help check if minimal pairs yield equal token lengths for the impossible language tokenizers. We expect the portion of unequal pairs to be 0.
To analyse all datasets (listed for convenience in `data_generation/generation_projects/impossible_blimp/master_dataset_list.txt`), run the bash script `data_generation/generation_projects/impossible_blimp/batch_analyse_token_lengths.sh`.

Part of sample output:

```bash
Analysing token lengths data_generation/outputs/impossible_blimp/v2/adjunct_island_20250623_165451%filtered.jsonl
Processing : 1000 sentences [00:00, 9574.05 sentences/s]
Portion of unequal pairs: 0/1000 
 Average difference: 0.0
Successfully analysed token lengths data_generation/outputs/impossible_blimp/v2/adjunct_island_20250623_165451%filtered.jsonl
----------------------------------------
```

### Sampling dataset:

To view sample sentences from the dataset, run the following command in the root project directory:

```bash
python -m data_generation.generation_projects.impossible_blimp.sample_dataset <path/to/dataset.jsonl> --samples <N> --seed <SEED>
```

For example:
```bash
python -m data_generation.generation_projects.impossible_blimp.sample_dataset data_generation/outputs/impossible_blimp/v2/anaphor_gender_agreement_20250618_113511%filtered.jsonl --samples 1 --seed 42
```

Sample output:

```bash
================================================================================
SAMPLE 1 (Index: 654)
================================================================================

ENGLISH:
  Good: All children go to one red school and cashiers go to at least as many.
  Bad:  All children go to one school and cashiers go to at least as many red.

OTHER VERSIONS:

SHUFFLE_NONDETERMINISTIC:
  Good: iers one as and school go to. least go many at red children cash toAll
  Bad:  iers andAll one cash. go many go least as to children to red at school

SHUFFLE_DETERMINISTIC21:
  Good:  oneiers and to as at schoolAll red children cash. many least go to go
  Bad:   one go cash at many least andAll school childreniers. red as to to go

SHUFFLE_LOCAL3:
  Good:  goAll children red to one cash school and toiers go as at least many.
  Bad:   goAll children school to oneiers and cash at go to many least as red.

SHUFFLE_LOCAL5:
  Good:  go one toAll children andiers cash red school at as least go to many.
  Bad:   go one toAll children cash goiers school and least many as to at red.

SHUFFLE_LOCAL10:
  Good:  one school go and to rediersAll cash children at as least. many go to
  Bad:   one and go cash to school goAlliers children least many as. red to at

SHUFFLE_EVEN_ODD:
  Good: All go one school cash go at as. children to red andiers to least many
  Bad:  All go one andiers to least many. children to school cash go at as red

REVERSE_PARTIAL:
  Good: All children go to one red school and cashiers go to at least as many.🅁
  Bad:  All children go to one school and cashiers go to at least as🅁. red many

REVERSE_FULL:
  Good: 🅁. many as least at to goiers cash and school red one to go childrenAll
  Bad:  . red many🅁 as least at to goiers cash and school one to go childrenAll

--------------------------------------------------------------------------------
```

### Analysing Performance and Perplexity:

Run the following command to generate accuracy vs perplexity scatterplot:

```
 python -m analysis.perplexity_scatterplot
```

The output can be found in ```analysis/output/accuracy_vs_perplexity_analysis.png```

## GUI

To make viewing experiment results easier, a GUI is provided in the `gui` directory.

To run the GUI, run the following command in the 'gui' project directory:

```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The GUI will be available at: http://localhost:8000/results

Note: The V1 version of the GUI is served at http://localhost:8000 and is deprecated.