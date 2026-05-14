import glob
import os

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
from tqdm import tqdm

from utils.impossible_utils import VALID_PERTURBATION_KEYS, GENRES

matplotlib.use("Agg")

BABYLM_FILES = [stem + ".train" for stem in GENRES]

rule all:
    input:
        "output/blimp/results.csv",
        "output/generation/scores/generation_scores_gpt2.csv"

#####################################
# EXPERIMENT 1: BLiMP Minimal Pairs #
# Step 1: Download BLiMP dataset    #
# Step 2: Filter for tok. issues    #
# Step 3: Apply perturbations       #
# Step 4: compare perplexities      #
#####################################

# Step 1: Download BLiMP dataset
checkpoint download_blimp:
    output:
        directory("data/blimp"),
        touch("data/blimp/.done")
    params:
        repo="https://github.com/alexwarstadt/blimp.git",
        branch="master",
    run:
        outdir = Path(output[0])

        # clone to temp. directory first, then copy to final destination
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)

            repo_dir = tmp / "repo"

            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth", "1",
                    "--branch", params.branch,
                    params.repo,
                    str(repo_dir),
                ],
                check=True,
            )

            source = repo_dir / "data"

            shutil.rmtree(outdir, ignore_errors=True)
            shutil.copytree(source, outdir)

def get_blimp_tasks():
    ckpt = checkpoints.download_blimp.get()

    tasks = [
        Path(f).stem
        for f in glob.glob(f"{ckpt.output[0]}/*.jsonl")
    ]

    return tasks

# Step 2: Filter for tokenization mismatches within minimal pairs.
# This is necessary to ensure that perturbations induce a "minimal" change.
rule filter_blimp_file:
    input:
        "data/blimp/{task}.jsonl"
    output:
        "data/blimp/filtered/{task}.jsonl"
    run:
        from minimal_pairs.filter_dataset import filter_dataset

        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)
        filter_dataset(input[0], output[0])

rule filter_blimp:
    input:
        "data/blimp/.done",
        lambda wildcards: expand("data/blimp/filtered/{task}.jsonl", task=get_blimp_tasks())
    output:
        touch("data/blimp/filtered/.done")

# Step 3: Apply perturbations to create BLiMP datasets for each perturbation type.
# Permutations are based on Kallini et al. (2024).
rule perturbed_blimp_file:
    input:
        "data/blimp/filtered/{task}.jsonl"
    output:
        "data/blimp/{perturbation}/{task}.jsonl"
    run:
        from minimal_pairs.modify_dataset import modify_dataset

        output_path = f"data/blimp/{wildcards.perturbation}/{wildcards.task}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        modify_dataset(input[0], wildcards.perturbation, output_path)

rule english_blimp_file:
    input:
        "data/blimp/filtered/{task}.jsonl"
    output:
        "data/blimp/english/{task}.jsonl"
    run:
        from minimal_pairs.modify_dataset import modify_dataset

        output_path = f"data/blimp/english/{wildcards.task}.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(input[0], output_path)

ruleorder: english_blimp_file > perturbed_blimp_file

rule perturbed_blimp:
    input:
        "data/blimp/filtered/.done",
        lambda wildcards: expand(
            "data/blimp/{perturbation}/{task}.jsonl", 
            perturbation=VALID_PERTURBATION_KEYS, 
            task=get_blimp_tasks()
        )
    output:
        touch("data/blimp/.done.perturbed")

rule evaluate_minimal_pairs:
    input:
        "data/blimp/.done.perturbed",
        lambda _: expand("data/blimp/{perturbation}/{task}.jsonl", task=get_blimp_tasks(),perturbation=VALID_PERTURBATION_KEYS)
    output:
        "output/blimp/results.csv"
    params:
        batch_size=config.get("batch_size", 16)
    run:
        from minimal_pairs.results import (
            ensure_results_csv_exists,
            MODEL_AND_LANGUAGE_OPTIONS,
            DEFAULT_MODEL_CHECKPOINT,
        )
        from minimal_pairs.evaluate import Evaluator
        from utils.impossible_utils import IMPOSSIBLE_MODEL_CHECKPOINTS
        ensure_results_csv_exists(output[0])
        for i in range(1, len(input) // len(VALID_PERTURBATION_KEYS) + 1):

            for j, perturbation in tqdm(enumerate(VALID_PERTURBATION_KEYS), desc="Evaluating perturbations"):
                evaluator = Evaluator(
                    dataset_path=input[(i-1)*len(VALID_PERTURBATION_KEYS) + j + 1],
                    model_name=perturbation,  # just use the first model for now
                    checkpoint="checkpoint-3000",
                    batch_size=params.batch_size,
                    results_csv=output[0],
                )
                results = evaluator.evaluate()
        

rule minimal_pairs_experiment:
    input:
        "output/blimp/results.csv"


#####################################
# EXPERIMENT 2: Generation quality  #
# Step 1: Generate & unperturb      #
# Step 2: Score w/ GPT-2            #
#####################################

# Step 1: Generate sentences, then unperturb to get "corrected" versions.
rule generate_lang:
    output:
        "output/generation/raw/{perturbation}.txt",
        "output/generation/corrected/{perturbation}.txt"
    params:
        num_gen_samples=config.get("num_gen_samples", 1000),
    run:
        from generation.generate import generate_samples

        generate_samples(wildcards.perturbation, params.num_gen_samples, output)

rule generate_all:
    input:
        expand("output/generation/{split}/{perturbation}.txt", split=["raw", "corrected"], perturbation=VALID_PERTURBATION_KEYS)

# Step 2: Score generated sentences using GPT-2 perplexity.
rule score_generation_file:
    input:
        "output/generation/corrected/{perturbation}.txt"
    output:
        "output/generation/scores/{perturbation}.csv"
    run:
        from generation.score import score_generation_file

        score_generation_file(wildcards.perturbation, input[0], output[0])

rule score_all_generations:
    input:
        expand("output/generation/scores/{perturbation}.csv", perturbation=VALID_PERTURBATION_KEYS)
    output:
        "output/generation/scores/generation_scores_gpt2.csv"
    run:
        import pandas as pd
        dfs = []
        for perturbation in VALID_PERTURBATION_KEYS:
            df = pd.read_csv(f"output/generation/scores/{perturbation}.csv")
            dfs.append(df)
        all_scores_df = pd.concat(dfs, ignore_index=True)
        all_scores_df.to_csv(output[0], index=False)



rule perplexity_boxplot:
    input:
        "output/generation/scores/generation_scores_gpt2.csv"
    output:
        "output/figures/perplexity_boxplot.pdf"
    run:
        from figures.perplexity_boxplot import plot_perplexity_boxplot
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)

        plot_perplexity_boxplot(input[0])

rule boxplot_stats:
    input:
        score_path="output/generation/scores/generation_scores_gpt2.csv",
        mlocal_path="output/mlocal_results.csv",
        perplexity_path="output/test_perplexity.csv"
    output:
        touch("output/figures/boxplot_stats.csv")
    run:
        from figures.perplexity_boxplot import boxplot_stats
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)

        boxplot_stats(input.score_path, input.mlocal_path, input.perplexity_path, output[0])

rule quantile_plot:
    input:
        "output/generation/scores/generation_scores_gpt2.csv"
    output:
        "output/figures/proportion_below_english_perplexity.pdf"
    run:
        from figures.quantiles import plot_quantiles
        os.makedirs(os.path.dirname(output[0]), exist_ok=True)

        plot_quantiles(input[0])


