# impossible-languages

## WORKFLOW
In this document, I outline the sequence of steps to set up and run the experiments for my project. Broadly, it is broken down into two stages: data generation and experiment running.

For detailed instructions on each step, please refer to the README.md file.

### Data generation

1. Generate/copy the base English dataset for a task
    1.1 The original BLiMP datasets can be found in ```data_generation/generation_projects/outputs/blimp```
    (these are copied from: https://github.com/alexwarstadt/blimp/tree/master/data)

    1.2 These files can be copied to ```data_generation/outputs/impossible_blimp/v2/``` (this folder is gitignored, so you may create it locally). 

    1.3 The dataset files names should be of the format ```<task_name>_<timestamp>.jsonl``` (to ensure uniqueness). This format is also critcal for scripts to function correctly, so I recommend using a small utitly to generate the timestamp and append it to the file name. Refer to ```data_generation/generation_projects/impossible_blimp/batch_rename.sh```.

    1.4 By copying the datasets, we can skip running the generation scripts

2. Update the ```data_generation/generation_projects/impossible_blimp/master_dataset_list.txt``` file list all the dataset filenames (one in each line)

    2.1 This will help when using batch_*.sh scripts to do batch processing of datasets

3. Ensure token length parity of the datasets

    3.1 Running the ```data_generation/generation_projects/impossible_blimp/filter_dataset.py``` script on a base dataset will result in a filtered version of the dataset being created in the same directory with the suffix ```%filtered```

    3.2 Leverage ```data_generation/generation_projects/impossible_blimp/batch_filter_dataset.sh``` to filter all datasets in the ```master_dataset_list.txt``` file

4. Update the ```data_generation/generation_projects/impossible_blimp/master_dataset_list.txt``` file list only the filtered dataset filenames (one in each line) 

    4.1 This will help when using batch_*.sh scripts to do batch processing of filtered datasets

5. Generate the impossible datasets

    5.1 Running the ```data_generation/generation_projects/impossible_blimp/modify_dataset.py``` script on a filtered dataset will result in an impossible version of the dataset being created in the same directory with the suffix ```<impossible_language_option>```

    5.2 Leverage ```data_generation/generation_projects/impossible_blimp/batch_generate.sh``` to generate all datasets in the ```master_dataset_list.txt``` file

### Running experiments
By now, you should have all your impossible datasets generated and located in ```data_generation/outputs/impossible_blimp/v2/```

The results of all experiments will be stored in ```experiments/output/v2/results.csv```

6. Run the experiment

    6.1 Running the ```experiments/experiment.py``` script on a dataset will result in an experiment being run on the dataset and the results being saved to the specified results csv file

    6.2 Leverage ```experiments/batch_run.sh``` to run all experiments for the datasets in one go. This script will also save the log output when run on the cluster. On the clsuter, this can be run as ```sbatch experiments/batch_run.sh```

### Visualising the results
By now, you should have all raw data saved in ```experiments/output/v2/results.csv```. Using the information under GUI in the README.md, you can visualise the results