import json

class ParallelEvaluationDataItem:
    """
    Represents a single data entry from the parallel evaluation dataset.
    Provides easy access to all fields as attributes.
    """
    def __init__(self, data_dict):
        for key, value in data_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        # A friendly representation for debugging
        return (f"ParallelEvaluationDataItem(pairID='{getattr(self, 'pairID', 'N/A')}', "
                f"field='{getattr(self, 'field', 'N/A')}', "
                f"linguistics_term='{getattr(self, 'linguistics_term', 'N/A')}')")

class ParallelEvaluationDatasetIterator:
    """
    An iterator for a file containing parallel evaluation dataset entries
    in JSON Lines format.

    Each yielded item is a ParallelEvaluationDataItem object, providing
    attribute-style access to its fields.
    """
    def __init__(self, filepath: str):
        """
        Initializes the iterator with the path to the dataset file.

        Args:
            filepath (str): The path to the JSON Lines file.
        """
        self.filepath = filepath

    def __iter__(self):
        """
        Makes the class itself an iterator.
        Opens the file and yields ParallelEvaluationDataItem objects line by line.
        """
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data_dict = json.loads(line.strip())
                    yield ParallelEvaluationDataItem(data_dict)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num} in '{self.filepath}': {e}")
                except Exception as e:
                    print(f"Warning: An unexpected error occurred on line {line_num} in '{self.filepath}': {e}")
        return self # This line is implicitly handled by the end of the generator function.


if __name__ == "__main__":
    data_filepath = "data_generation/outputs/impossible_blimp/english_to_local_shuffle_three_for_anaphor_agreement_gender.jsonl"
  
    # Initialize the iterator
    dataset_iter = ParallelEvaluationDatasetIterator(data_filepath)

    print("Iterating through the dataset:")
    count = 0
    for i, item in enumerate(dataset_iter):
        print(f"\n--- Item {i+1} ---")
        print(f"Dataset A Good: {item.dataset_A_grammatical}")
        print(f"Dataset A Bad: {item.dataset_A_ungrammatical}")
        print(f"Dataset B Good: {item.dataset_B_grammatical}")
        print(f"Dataset B Bad: {item.dataset_B_ungrammatical}")
        print(f"Linguistics Term: {item.linguistics_term}")
        print(f"Pair ID: {item.pairID}")

        # Example of checking boolean flags
        if item.simple_LM_method:
            print("  (Uses Simple LM Method)")
        if hasattr(item, 'non_existent_attribute'): # How to check if an attribute exists
            print("This won't print.")

        count += 1
        if count >= 5:  # Limit to first 5 items for brevity
            break