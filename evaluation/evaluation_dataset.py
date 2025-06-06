import json

class ParallelEvaluationDataItem:
    """
    Represents a single data entry from the parallel evaluation dataset.
    Provides easy access to all fields as attributes.
    """
    def __init__(self, data_dict):
        self.dataset_A_grammatical = data_dict["dataset_A_grammatical"]
        self.dataset_A_ungrammatical = data_dict["dataset_A_ungrammatical"]
        self.one_prefix_prefix = data_dict["one_prefix_prefix"]
        self.one_prefix_word_good = data_dict["one_prefix_word_good"]
        self.one_prefix_word_bad = data_dict["one_prefix_word_bad"]
        self.dataset_B_grammatical = data_dict["dataset_B_grammatical"]
        self.dataset_B_ungrammatical = data_dict["dataset_B_ungrammatical"]
        self.field = data_dict["field"]
        self.linguistics_term = data_dict["linguistics_term"]
        self.UID = data_dict["UID"]
        self.simple_LM_method = data_dict["simple_LM_method"]
        self.one_prefix_method = data_dict["one_prefix_method"]
        self.two_prefix_method = data_dict["two_prefix_method"]
        self.lexically_identical = data_dict["lexically_identical"]
        self.pairID = data_dict["pairID"]

    def __repr__(self):
        # A friendly representation for debugging
        return (f"ParallelEvaluationDataItem(pairID='{self.pairID}', "
                f"field='{self.field}', "
                f"linguistics_term='{self.linguistics_term}')")

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


class ParallelDataBatchLoader:
    """
    A utility class to efficiently load and batch parallel evaluation data.
    """
    def __init__(self, filepath: str, batch_size: int = 16):
        self.filepath = filepath
        self.batch_size = batch_size
    
    def __iter__(self):
        """
        Yields batches of ParallelEvaluationDataItem objects
        """
        items_batch = []
        for item in ParallelEvaluationDatasetIterator(self.filepath):
            items_batch.append(item)
            
            if len(items_batch) >= self.batch_size:
                yield items_batch
                items_batch = []
        
        # Yield any remaining items
        if items_batch:
            yield items_batch


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