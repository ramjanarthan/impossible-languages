import json

class EvaluationDataItem:
    """
    Represents a single data entry from the new parallel evaluation dataset schema.
    Accessible fields: sentence_good, sentence_bad, UID, pairID
    """
    def __init__(self, data_dict):
        self.sentence_good = data_dict["sentence_good"]
        self.sentence_bad = data_dict["sentence_bad"]
        self.UID = data_dict["UID"]
        self.pairID = data_dict["pairID"]

    def __repr__(self):
        return (f"EvaluationDataItem(pairID='{self.pairID}', UID='{self.UID}', "
                f"sentence_good='{self.sentence_good[:30]}...', sentence_bad='{self.sentence_bad[:30]}...')")

class EvaluationDatasetIterator:
    """
    An iterator for a file containing parallel evaluation dataset entries
    in JSON Lines format.

    Each yielded item is a EvaluationDataItem object, providing
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
                    yield EvaluationDataItem(data_dict)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed JSON on line {line_num} in '{self.filepath}': {e}")
                except Exception as e:
                    print(f"Warning: An unexpected error occurred on line {line_num} in '{self.filepath}': {e}")
        return self # This line is implicitly handled by the end of the generator function.


class DataBatchLoader:
    """
    A utility class to efficiently load and batch evaluation data.
    """ 
    def __init__(self, filepath: str, batch_size: int = 16):
        self.filepath = filepath
        self.batch_size = batch_size
    
    def __iter__(self):
        """
        Yields batches of EvaluationDataItem objects
        """
        items_batch = []
        for item in EvaluationDatasetIterator(self.filepath):
            items_batch.append(item)
            
            if len(items_batch) >= self.batch_size:
                yield items_batch
                items_batch = []
        
        # Yield any remaining items
        if items_batch:
            yield items_batch


if __name__ == "__main__":
    data_filepath = "data_generation/outputs/impossible_blimp/v2/distractor_agreement_relative_clause_20250616_174118%reverse_full.jsonl"
  
    # Initialize the iterator
    dataset_iter = EvaluationDatasetIterator(data_filepath)

    print("Iterating through the dataset:")
    count = 0
    for i, item in enumerate(dataset_iter):
        print(f"\n--- Item {i+1} ---")
        print(f"Sentence Good: {item.sentence_good}")
        print(f"Sentence Bad: {item.sentence_bad}")
        print(f"Pair ID: {item.pairID}")

        count += 1
        if count >= 5:  # Limit to first 5 items for brevity
            break