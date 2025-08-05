import json
import random
import os

def sample_sentences_from_datasets(dataset_list_path, output_path, num_samples=100):
    """
    Samples sentences from a list of JSONL dataset files.

    Args:
        dataset_list_path (str): Path to a file containing a list of dataset file paths.
        output_path (str): Path to write the sampled sentences to.
        num_samples (int): The number of sentences to sample.
    """
    all_good_sentences = []
    base_dir = os.path.dirname(dataset_list_path) # Assuming paths in the list are relative to the list's location
    project_root = '/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages'

    print(f"Reading dataset list from: {dataset_list_path}")

    try:
        with open(dataset_list_path, 'r') as f:
            dataset_files = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Dataset list file not found at {dataset_list_path}")
        return

    for file_path in dataset_files:
        # Construct the full path relative to the project root
        full_path = os.path.join(project_root, file_path)
        
        try:
            with open(full_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'sentence_good' in data:
                            all_good_sentences.append(data['sentence_good'])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from a line in {full_path}")
        except FileNotFoundError:
            print(f"Warning: Dataset file not found: {full_path}")

    if not all_good_sentences:
        print("No 'sentence_good' entries found in any dataset.")
        return

    print(f"Collected a total of {len(all_good_sentences)} sentences.")

    # Sample the sentences
    if len(all_good_sentences) < num_samples:
        print(f"Warning: Number of available sentences ({len(all_good_sentences)}) is less than requested samples ({num_samples}). Using all available sentences.")
        sampled_sentences = all_good_sentences
    else:
        sampled_sentences = random.sample(all_good_sentences, num_samples)

    # Write to output file
    try:
        with open(output_path, 'w') as f:
            for sentence in sampled_sentences:
                f.write(sentence + '\n')
        print(f"Successfully wrote {len(sampled_sentences)} sentences to {output_path}")
    except IOError as e:
        print(f"Error writing to output file {output_path}: {e}")


if __name__ == "__main__":
    MASTER_LIST_PATH = 'data_generation/generation_projects/impossible_blimp/master_dataset_list.txt'
    OUTPUT_SENTENCES_PATH = 'analysis/sample_sentences.txt'
    
    # Get the absolute path for the master list
    project_root = '/Users/ramjanarthan/Desktop/UoE/sem_2/ipp/impossible-languages'
    absolute_master_list_path = os.path.join(project_root, MASTER_LIST_PATH)
    absolute_output_path = os.path.join(project_root, OUTPUT_SENTENCES_PATH)

    sample_sentences_from_datasets(absolute_master_list_path, absolute_output_path, num_samples=100)
