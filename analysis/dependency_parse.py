import spacy, json
from spacy import displacy
from typing import Dict, List, Tuple, Any
from analysis.impossible_dependency_parse import create_perturbed_doc
from analysis.utils import create_spacy_perturbation_map
from data_generation.utils.impossible_utils import PERTURBATIONS_PAIR

# Load the English language model
nlp = spacy.load('en_core_web_sm')

# Example sentence to parse
sentence = "Timothy didn't boast about himself."
reverse_sentence = ". himself about🅁 boast't didnothyTim"
reverse_parial_sentence = "Timothy didn't boast🅁. himself about"
shuffle_det_sentence = "aboutTim boastothy himself't. didn"
shuffle_non_det_sentence = ". himself didn about boastTimothy't"
shuffle_local3 = "didnTimothy about't boast himself."
shuffle_odd_even = "Tim didn boast himselfothy't about."

# Parse the sentence using spaCy
doc = nlp(sentence)

# Visualize the dependency tree with default settings
# displacy.serve(doc, style='dep', port=5004)

def get_dependency_stats(doc) -> Dict[str, Any]:
    """
    Analyze dependency parse statistics from a spaCy Doc object.
    
    Args:
        doc: spaCy Doc object containing dependency parse information
        
    Returns:
        Dict containing:
        - is_projective: Whether the dependency tree is projective
        - total_dependency_distance: Sum of head-to-dependent distances
        - normalized_dependency_distance: Token-normalized version of total distance
        - same_word_token_distances: Distances between tokens of the same word
        - crossing_dependencies_count: Number of crossing dependencies
    """
    
    # Filter out punctuation and spaces for cleaner analysis
    tokens = [token for token in doc if not token.is_punct and not token.is_space]
    n_tokens = len(tokens)
    
    if n_tokens == 0:
        return {
            "is_projective": True,
            "total_dependency_distance": 0,
            "normalized_dependency_distance": 0.0,
            "same_word_token_distances": [],
            "crossing_dependencies_count": 0
        }
    
    # Create mapping from token to index for easier processing
    token_to_idx = {token.i: idx for idx, token in enumerate(tokens)}
    
    # 1. Check if tree is projective and count crossing dependencies
    is_projective, crossing_count = check_projectivity_and_crossings(tokens, token_to_idx)
    
    # 2. Calculate total dependency distance
    total_distance = calculate_total_dependency_distance(tokens, token_to_idx)
    
    # 3. Calculate normalized dependency distance
    normalized_distance = total_distance / n_tokens if n_tokens > 0 else 0.0
    
    # 4. Calculate distances between tokens of the same word
    same_word_distances = calculate_same_word_distances(doc)
    
    return {
        "is_projective": is_projective,
        "total_dependency_distance": total_distance,
        "normalized_dependency_distance": round(normalized_distance, 4),
        "same_word_token_distances": same_word_distances,
        "crossing_dependencies_count": crossing_count
    }


def check_projectivity_and_crossings(tokens: List, token_to_idx: Dict[int, int]) -> Tuple[bool, int]:
    """
    Check if the dependency tree is projective and count crossing dependencies.
    
    A dependency tree is projective if no two dependencies cross each other.
    """
    dependencies = []
    
    # Extract all dependencies with their positions
    for token in tokens:
        if token.head != token:  # Not root
            head_idx = token_to_idx.get(token.head.i)
            dep_idx = token_to_idx.get(token.i)
            
            if head_idx is not None and dep_idx is not None:
                # Store as (min_pos, max_pos) for easier crossing detection
                min_pos = min(head_idx, dep_idx)
                max_pos = max(head_idx, dep_idx)
                dependencies.append((min_pos, max_pos, token.dep_, token.text))
    
    # Count crossings
    crossing_count = 0
    for i, (min1, max1, _, _) in enumerate(dependencies):
        for j, (min2, max2, _, _) in enumerate(dependencies[i+1:], i+1):
            # Two dependencies cross if one starts before the other but ends after it starts
            if (min1 < min2 < max1 < max2) or (min2 < min1 < max2 < max1):
                crossing_count += 1
    
    is_projective = crossing_count == 0
    
    return is_projective, crossing_count


def calculate_total_dependency_distance(tokens: List, token_to_idx: Dict[int, int]) -> int:
    """
    Calculate the sum of distances between heads and their dependents.
    """
    total_distance = 0
    
    for token in tokens:
        if token.head != token:  # Not root
            head_idx = token_to_idx.get(token.head.i)
            dep_idx = token_to_idx.get(token.i)
            
            if head_idx is not None and dep_idx is not None:
                distance = abs(head_idx - dep_idx)
                total_distance += distance
    
    return total_distance


def calculate_same_word_distances(doc) -> List[Dict[str, Any]]:
    """
    Calculate distances between tokens that belong to the same original word.
    This handles cases where a single word gets split into multiple tokens
    (e.g., contractions like "don't" -> "do" + "n't", or compound words).
    """
    same_word_distances = []
    
    # Group tokens by their idx (word index in the original text)
    # Tokens that share the same idx belong to the same original word
    word_groups = {}
    
    for token in doc:
        if not token.is_punct and not token.is_space:
            # Use token.idx (character offset) to group tokens from the same word
            # But we need to be more sophisticated - check if tokens are part of 
            # multi-token words using spaCy's token attributes
            
            # Check if this token is part of a multi-token word
            if hasattr(token, 'norm_') and token.norm_ != token.text:
                # This might be part of a normalized form
                norm_key = f"{token.idx}_{token.norm_}"
            else:
                # For contractions and multi-token words, we can use the head token's idx
                # or check if consecutive tokens have no whitespace between them
                norm_key = token.idx
            
            if norm_key not in word_groups:
                word_groups[norm_key] = []
            word_groups[norm_key].append(token)
    
    # More robust approach: check for tokens that are adjacent in the original text
    # but may be separated in the token sequence
    processed_tokens = {}
    
    for i, token in enumerate(doc):
        if token.is_punct or token.is_space:
            continue
            
        # Check if this token is part of a multi-word token by looking at character positions
        current_end = token.idx + len(token.text)
        
        # Look for other tokens that start where this one ends (no space between)
        for j, other_token in enumerate(doc):
            if i != j and not other_token.is_punct and not other_token.is_space:
                # Check if tokens are adjacent in the original text
                if (other_token.idx == current_end or 
                    token.idx == other_token.idx + len(other_token.text)):
                    
                    # These tokens are part of the same original word
                    word_key = min(token.idx, other_token.idx)
                    
                    if word_key not in processed_tokens:
                        processed_tokens[word_key] = []
                    
                    if token not in processed_tokens[word_key]:
                        processed_tokens[word_key].append(token)
                    if other_token not in processed_tokens[word_key]:
                        processed_tokens[word_key].append(other_token)
    
    # Calculate distances for multi-token words
    for word_start_idx, token_list in processed_tokens.items():
        if len(token_list) > 1:
            # Sort tokens by their position in the document
            token_list.sort(key=lambda t: t.i)
            
            # Calculate distances between all pairs within the same word
            # Only count distance if there are other tokens in between
            for i in range(len(token_list)):
                for j in range(i + 1, len(token_list)):
                    token_distance = abs(token_list[i].i - token_list[j].i)
                    
                    # Only add to results if tokens are NOT adjacent (distance > 1)
                    if token_distance > 1:
                        original_word = doc.text[word_start_idx:token_list[-1].idx + len(token_list[-1].text)]
                        
                        same_word_distances.append({
                            "original_word": original_word,
                            "token_positions": [token_list[i].i, token_list[j].i],
                            "distance": token_distance,
                            "tokens": [token_list[i].text, token_list[j].text],
                            "char_start": word_start_idx
                        })
    
    return same_word_distances


# Example usage and testing function
def test_dependency_stats():
    """
    Test function to demonstrate usage with spaCy.
    """
    try:        
        nlp = spacy.load("en_core_web_sm")
        
        # Test with original and perturbed sentences
        sentence = "Timothy didn't boast about himself."
        reverse_sentence = ". himself about🅁 boast't didnothyTim"
        reverse_parial_sentence = "Timothy didn't boast🅁. himself about"
        shuffle_det_sentence = "aboutTim boastothy himself't. didn"
        shuffle_non_det_sentence = ". himself didn about boastTimothy't"
        shuffle_local3 = "didnTimothy about't boast himself."
        shuffle_odd_even = "Tim didn boast himselfothy't about."
        
        # print(f"Original: '{sentence}'")
        original_doc = nlp(sentence)

        tokenizer = PERTURBATIONS_PAIR["reverse_full"]["gpt2_tokenizer"]
        gpt2_tokens = tokenizer.encode(sentence)
        for token in gpt2_tokens:
            print(tokenizer.decode(token))
        # print(gpt2_tokens)

        # print(dir(original_doc))
        # pprint.pprint(original_doc)            

        for token in original_doc:
            print(token.text, token.dep_, token.head.text, token.head.i)            

        # original_stats = get_dependency_stats(original_doc)
        # print("Original stats:")
        # print(json.dumps(original_stats, indent=2))
        
        # print(f"\n Reversed: '{reverse_sentence}'")
                # Generate the perturbation map using the new alignment-based method
        perturbation_map = create_spacy_perturbation_map(original_doc, tokenizer, "reverse_full")

        perturbed_doc = create_perturbed_doc(original_doc, perturbation_map)

        print("-----")
        for token in perturbed_doc:
            print(token.text, token.dep_, token.head.text, token.head.i)

        displacy.serve(perturbed_doc, style='dep', port=5004)


        # perturbed_stats = get_dependency_stats(perturbed_doc)
        # print("Perturbed stats:")
        # print(json.dumps(perturbed_stats, indent=2))

        # print(f"\n Shuffled: '{shuffle_det_sentence}'")
        # perturbed_doc_shuffle = create_perturbed_doc(sentence, original_doc, shuffle_det_sentence)
        # perturbed_stats_shuffle = get_dependency_stats(perturbed_doc_shuffle)
        # print("Perturbed stats:")
        # print(json.dumps(perturbed_stats_shuffle, indent=2))

        # print(f"\n Shuffled: '{shuffle_odd_even}'")
        # perturbed_doc_shuffle_odd_e = create_perturbed_doc(sentence, original_doc, shuffle_odd_even)
        # perturbed_stats_shuffle_odd_e = get_dependency_stats(perturbed_doc_shuffle_odd_e)
        # print("Perturbed stats:")
        # print(json.dumps(perturbed_stats_shuffle_odd_e, indent=2))

            
    except ImportError:
        print("spaCy not installed. Install with: pip install spacy")
        print("Also download a model with: python -m spacy download en_core_web_sm")


if __name__ == "__main__":
    test_dependency_stats()