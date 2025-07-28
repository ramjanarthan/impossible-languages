import spacy
from typing import List, Dict, Any, Tuple, Set
from numpy.random import default_rng
from data_generation.utils.impossible_utils import PERTURBATIONS
import statistics

def align_tokens_with_tokenizer(sentence: str, original_doc: spacy.tokens.Doc, tokenizer) -> List[Dict[str, Any]]:
    """
    Align spaCy tokens with a given tokenizer while preserving dependency relationships.
    
    Args:
        sentence: Original sentence string
        original_doc: spaCy Doc object from parsing the sentence
        tokenizer: Tokenizer object (e.g., GPT-2 tokenizer) with encode/decode methods
    
    Returns:
        List of token dictionaries with aligned tokens and updated dependencies
    """
    
    # Get tokenizer tokens and their text representations
    tokenizer_ids = tokenizer.encode(sentence)
    tokenizer_tokens = [tokenizer.decode([token_id]) for token_id in tokenizer_ids]
    
    # Create mapping from character positions to spaCy tokens
    char_to_spacy_token = {}
    for i, token in enumerate(original_doc):
        for char_idx in range(token.idx, token.idx + len(token.text)):
            char_to_spacy_token[char_idx] = i
    
    # Align tokenizer tokens with spaCy tokens
    tokenizer_to_spacy_mapping = []
    current_char_pos = 0
    
    for tok_token in tokenizer_tokens:
        # Clean the token (remove leading/trailing spaces for matching)
        clean_token = tok_token.strip()
        if not clean_token:
            tokenizer_to_spacy_mapping.append(None)
            continue
            
        # Find where this tokenizer token appears in the sentence
        # Try both with and without leading space
        token_start = -1
        search_variants = [tok_token, clean_token]
        
        for variant in search_variants:
            token_start = sentence.find(variant, current_char_pos)
            if token_start != -1:
                tok_token = variant
                break
        
        if token_start == -1:
            # Handle special tokens or encoding issues
            tokenizer_to_spacy_mapping.append(None)
            continue
            
        token_end = token_start + len(tok_token)
        
        # Find which spaCy token(s) this tokenizer token corresponds to
        spacy_tokens_covered = set()
        for char_idx in range(token_start, token_end):
            if char_idx in char_to_spacy_token:
                spacy_tokens_covered.add(char_to_spacy_token[char_idx])
        
        if spacy_tokens_covered:
            # Use the first spaCy token as the primary alignment
            primary_spacy_token = min(spacy_tokens_covered)
            tokenizer_to_spacy_mapping.append((primary_spacy_token, spacy_tokens_covered))
        else:
            tokenizer_to_spacy_mapping.append(None)
        
        current_char_pos = token_end
    
    # Create mapping from old spaCy token indices to new token indices
    spacy_to_new_token_mapping = {}
    new_token_groups = {}  # Groups of new tokens that belong to same original token
    
    for new_idx, alignment in enumerate(tokenizer_to_spacy_mapping):
        if alignment is not None:
            if isinstance(alignment, tuple):
                primary_spacy_idx, spacy_tokens_covered = alignment
            else:
                primary_spacy_idx = alignment
                spacy_tokens_covered = {alignment}
            
            # Map the primary spaCy token to this new token
            if primary_spacy_idx not in spacy_to_new_token_mapping:
                spacy_to_new_token_mapping[primary_spacy_idx] = new_idx
            
            # Track which new tokens belong to the same original spaCy token
            for spacy_idx in spacy_tokens_covered:
                if spacy_idx not in new_token_groups:
                    new_token_groups[spacy_idx] = []
                new_token_groups[spacy_idx].append(new_idx)
    
    # Create list of token dictionaries
    aligned_tokens = []
    
    for new_idx, tok_token in enumerate(tokenizer_tokens):
        alignment = tokenizer_to_spacy_mapping[new_idx]
        
        token_dict = {
            'index': new_idx,
            'text': tok_token,
            'head_index': new_idx,  # Default to self
            'dep': 'ROOT',  # Default dependency
            'pos': '',
            'tag': '',
            'lemma': tok_token,
            'ent_type': '',
            'original_index': new_idx  # Track original position for perturbations
        }
        
        # Copy attributes from aligned spaCy token
        if alignment is not None:
            if isinstance(alignment, tuple):
                primary_spacy_idx, _ = alignment
            else:
                primary_spacy_idx = alignment
            
            original_token = original_doc[primary_spacy_idx]
            token_dict.update({
                'pos': original_token.pos_,
                'tag': original_token.tag_,
                'lemma': original_token.lemma_,
                'ent_type': original_token.ent_type_
            })
        
        aligned_tokens.append(token_dict)
    
    # Update dependency relationships
    # First pass: map regular dependencies
    for old_idx, old_token in enumerate(original_doc):
        if old_idx in spacy_to_new_token_mapping:
            new_idx = spacy_to_new_token_mapping[old_idx]
            
            # Map head relationship
            if old_token.head.i == old_idx:  # Root token
                aligned_tokens[new_idx]['head_index'] = new_idx
                aligned_tokens[new_idx]['dep'] = old_token.dep_
            elif old_token.head.i in spacy_to_new_token_mapping:
                head_new_idx = spacy_to_new_token_mapping[old_token.head.i]
                aligned_tokens[new_idx]['head_index'] = head_new_idx
                aligned_tokens[new_idx]['dep'] = old_token.dep_
    
    # Second pass: handle "linked" relationships within word groups
    for spacy_idx, new_token_indices in new_token_groups.items():
        if len(new_token_indices) > 1:
            # Sort by position
            new_token_indices.sort()
            
            # Find the leftmost token that will be the head
            leftmost_idx = new_token_indices[0]
            
            # If this group has a dependency head, assign it to the leftmost token
            if spacy_idx < len(original_doc):
                old_token = original_doc[spacy_idx]
                if old_token.head.i == spacy_idx:  # Root
                    aligned_tokens[leftmost_idx]['head_index'] = leftmost_idx
                    aligned_tokens[leftmost_idx]['dep'] = old_token.dep_
                elif old_token.head.i in spacy_to_new_token_mapping:
                    head_new_idx = spacy_to_new_token_mapping[old_token.head.i]
                    aligned_tokens[leftmost_idx]['head_index'] = head_new_idx
                    aligned_tokens[leftmost_idx]['dep'] = old_token.dep_
            
            # Link other tokens in the group to the leftmost token
            for token_idx in new_token_indices[1:]:
                aligned_tokens[token_idx]['head_index'] = leftmost_idx
                aligned_tokens[token_idx]['dep'] = 'linked'
    
    return aligned_tokens


def apply_windowed_shuffle_perturbation(tokens: List[Dict[str, Any]], window: int, seed: int) -> List[Dict[str, Any]]:
    """
    Apply windowed shuffle perturbation while preserving dependency relationships.
    
    Args:
        tokens: List of token dictionaries
        window: Window size for shuffling
        seed: Random seed for reproducibility
    
    Returns:
        New list of tokens with updated indices and dependency relationships
    """
    
    # Step 1: Apply the perturbation to get new ordering
    shuffled_tokens = []
    for i in range(0, len(tokens), window):
        batch = tokens[i:i+window].copy()
        default_rng(seed).shuffle(batch)
        shuffled_tokens += batch
    
    # Step 2: Create mapping from old indices to new indices
    old_to_new_index = {}
    for new_idx, token in enumerate(shuffled_tokens):
        old_idx = token['original_index']
        old_to_new_index[old_idx] = new_idx
    
    # Step 3: Update indices and head relationships
    updated_tokens = []
    for new_idx, token in enumerate(shuffled_tokens):
        # Create new token with updated index
        updated_token = token.copy()
        updated_token['index'] = new_idx
        
        # Update head_index using the mapping
        old_head_idx = token['head_index']
        if old_head_idx in old_to_new_index:
            updated_token['head_index'] = old_to_new_index[old_head_idx]
        else:
            # Fallback: if head not found, point to self (shouldn't happen in normal cases)
            updated_token['head_index'] = new_idx
            print(f"Warning: Head index {old_head_idx} not found for token {token['text']}")
        
        updated_tokens.append(updated_token)
    
    return updated_tokens


def apply_reverse_perturbation(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply reverse perturbation while preserving dependency relationships.
    
    Args:
        tokens: List of token dictionaries
    
    Returns:
        New list of tokens with updated indices and dependency relationships
    """
    
    # Step 1: Reverse the token order
    reversed_tokens = tokens[::-1]
    
    # Step 2: Create mapping from old indices to new indices
    old_to_new_index = {}
    for new_idx, token in enumerate(reversed_tokens):
        old_idx = token['original_index']
        old_to_new_index[old_idx] = new_idx
    
    # Step 3: Update indices and head relationships
    updated_tokens = []
    for new_idx, token in enumerate(reversed_tokens):
        # Create new token with updated index
        updated_token = token.copy()
        updated_token['index'] = new_idx
        
        # Update head_index using the mapping
        old_head_idx = token['head_index']
        if old_head_idx in old_to_new_index:
            updated_token['head_index'] = old_to_new_index[old_head_idx]
        else:
            # Fallback: if head not found, point to self
            updated_token['head_index'] = new_idx
            print(f"Warning: Head index {old_head_idx} not found for token {token['text']}")
        
        updated_tokens.append(updated_token)
    
    return updated_tokens


def print_aligned_tokens(tokens: List[Dict[str, Any]], title: str = "Aligned Tokens"):
    """
    Print aligned tokens in a nice tabular format.
    """
    print(f"\n{title}:")
    print("=" * 80)
    
    # Calculate column widths
    max_idx_width = max(len(str(token['index'])) for token in tokens)
    max_text_width = max(len(repr(token['text'])) for token in tokens)
    max_dep_width = max(len(token['dep']) for token in tokens)
    max_head_text_width = max(len(repr(tokens[token['head_index']]['text'])) for token in tokens)
    max_head_idx_width = max(len(str(token['head_index'])) for token in tokens)
    max_pos_width = max(len(token['pos']) for token in tokens) if any(token['pos'] for token in tokens) else 3
    
    # Ensure minimum widths
    idx_width = max(max_idx_width, 3)
    text_width = max(max_text_width, 8)
    dep_width = max(max_dep_width, 8)
    head_text_width = max(max_head_text_width, 8)
    head_idx_width = max(max_head_idx_width, 8)
    pos_width = max(max_pos_width, 3)
    
    # Print header
    header = f"{'Idx':<{idx_width}} {'Token':<{text_width}} {'Dep':<{dep_width}} {'Head Token':<{head_text_width}} {'Head Idx':<{head_idx_width}} {'POS':<{pos_width}}"
    print(header)
    print("-" * len(header))
    
    # Print tokens
    for token in tokens:
        head_token_text = repr(tokens[token['head_index']]['text'])
        
        row = (f"{token['index']:<{idx_width}} "
               f"{repr(token['text']):<{text_width}} "
               f"{token['dep']:<{dep_width}} "
               f"{head_token_text:<{head_text_width}} "
               f"{token['head_index']:<{head_idx_width}} "
               f"{token['pos']:<{pos_width}}")
        print(row)
    
    print()


def print_aligned_tokens_simple(tokens: List[Dict[str, Any]], title: str = "Aligned Tokens"):
    """
    Print aligned tokens in a simpler format.
    """
    print(f"\n{title}:")
    
    for token in tokens:
        head_token = tokens[token['head_index']]
        print(f"{token['index']:<3} {token['text']:<15} {token['dep']:<12} {head_token['text']:<15} {token['head_index']}")

def calculate_dependency_statistics(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a dependency parse.
    
    Args:
        tokens: List of token dictionaries with 'index', 'head_index', 'dep', etc.
    
    Returns:
        Dictionary containing all dependency statistics
    """
    
    # Calculate individual statistics
    is_proj = is_projective(tokens)
    total_dist = total_dependency_distance(tokens)
    norm_dist = normalized_dependency_distance(tokens)
    same_word_dists = same_word_token_distances(tokens)
    crossing_count = crossing_dependencies_count(tokens)
    
    return {
        'is_projective': is_proj,
        'total_dependency_distance': total_dist,
        'normalized_dependency_distance': norm_dist,
        'same_word_token_distances': same_word_dists,
        'crossing_dependencies_count': crossing_count,
        'num_tokens': len(tokens),
        'average_dependency_distance': total_dist / len(tokens) if tokens else 0
    }


def is_projective(tokens: List[Dict[str, Any]]) -> bool:
    """
    Check if the dependency tree is projective.
    
    A dependency tree is projective if there are no crossing dependencies.
    This means that for any dependency arc (h, d), all tokens between h and d
    must be descendants of h.
    
    Args:
        tokens: List of token dictionaries
    
    Returns:
        True if projective, False otherwise
    """
    
    def get_descendants(head_idx: int, tokens: List[Dict[str, Any]]) -> Set[int]:
        """Get all descendants (direct and indirect) of a head token."""
        descendants = set()
        direct_children = [i for i, token in enumerate(tokens) 
                          if token['head_index'] == head_idx and i != head_idx]
        
        for child in direct_children:
            descendants.add(child)
            descendants.update(get_descendants(child, tokens))
        
        return descendants
    
    # Check each dependency arc
    for token in tokens:
        head_idx = token['head_index']
        dep_idx = token['index']
        
        # Skip self-loops (ROOT dependencies)
        if head_idx == dep_idx:
            continue
            
        # Get the span between head and dependent
        start_idx = min(head_idx, dep_idx)
        end_idx = max(head_idx, dep_idx)
        
        # Get all descendants of the head
        descendants = get_descendants(head_idx, tokens)
        descendants.add(head_idx)  # Include the head itself
        
        # Check if all tokens in the span are descendants of head
        for i in range(start_idx, end_idx + 1):
            if i not in descendants:
                return False
    
    return True


def total_dependency_distance(tokens: List[Dict[str, Any]]) -> int:
    """
    Calculate the sum of all head-to-dependent distances.
    
    Args:
        tokens: List of token dictionaries
    
    Returns:
        Sum of linear distances between heads and dependents
    """
    total_distance = 0
    
    for token in tokens:
        head_idx = token['head_index']
        dep_idx = token['index']
        
        # Calculate linear distance
        distance = abs(head_idx - dep_idx)
        total_distance += distance
    
    return total_distance


def normalized_dependency_distance(tokens: List[Dict[str, Any]]) -> float:
    """
    Calculate token-normalized dependency distance.
    
    Args:
        tokens: List of token dictionaries
    
    Returns:
        Total dependency distance normalized by number of tokens
    """
    if not tokens:
        return 0.0
    
    total_dist = total_dependency_distance(tokens)
    return total_dist / len(tokens)


def same_word_token_distances(tokens: List[Dict[str, Any]]) -> List[int]:
    """
    Calculate distances between tokens that belong to the same word.
    Based on 'linked' dependencies which connect sub-word tokens.
    
    Args:
        tokens: List of token dictionaries
    
    Returns:
        List of distances between linked tokens
    """
    distances = []
    
    # Find all 'linked' dependencies
    for token in tokens:
        if token['dep'] == 'linked':
            head_idx = token['head_index']
            dep_idx = token['index']
            distance = abs(head_idx - dep_idx)
            distances.append(distance)
    
    return distances


def crossing_dependencies_count(tokens: List[Dict[str, Any]]) -> int:
    """
    Count the number of crossing dependencies.
    
    Two dependencies (h1, d1) and (h2, d2) cross if:
    - h1 < h2 < d1 < d2, or
    - h2 < h1 < d2 < d1
    
    Args:
        tokens: List of token dictionaries
    
    Returns:
        Number of crossing dependency pairs
    """
    
    # Get all dependency arcs (excluding self-loops)
    arcs = []
    for token in tokens:
        head_idx = token['head_index']
        dep_idx = token['index']
        
        if head_idx != dep_idx:  # Skip self-loops (ROOT)
            # Store arc as (start, end) where start < end
            start = min(head_idx, dep_idx)
            end = max(head_idx, dep_idx)
            arcs.append((start, end))
    
    # Count crossing pairs
    crossing_count = 0
    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            arc1 = arcs[i]
            arc2 = arcs[j]
            
            # Check if arcs cross
            if arcs_cross(arc1, arc2):
                crossing_count += 1
    
    return crossing_count


def arcs_cross(arc1: Tuple[int, int], arc2: Tuple[int, int]) -> bool:
    """
    Check if two dependency arcs cross.
    
    Args:
        arc1: First arc as (start, end) tuple
        arc2: Second arc as (start, end) tuple
    
    Returns:
        True if arcs cross, False otherwise
    """
    start1, end1 = arc1
    start2, end2 = arc2
    
    # Two arcs cross if one starts inside the other but ends outside
    # Pattern: start1 < start2 < end1 < end2 or start2 < start1 < end2 < end1
    return (start1 < start2 < end1 < end2) or (start2 < start1 < end2 < end1)


def print_dependency_statistics(tokens: List[Dict[str, Any]], title: str = "Dependency Statistics"):
    """
    Print comprehensive dependency statistics in a readable format.
    
    Args:
        tokens: List of token dictionaries
        title: Title for the output
    """
    stats = calculate_dependency_statistics(tokens)
    
    print(f"\n{title}:")
    print("=" * 50)
    print(f"Number of tokens: {stats['num_tokens']}")
    print(f"Is projective: {stats['is_projective']}")
    print(f"Total dependency distance: {stats['total_dependency_distance']}")
    print(f"Normalized dependency distance: {stats['normalized_dependency_distance']:.3f}")
    print(f"Average dependency distance: {stats['average_dependency_distance']:.3f}")
    print(f"Crossing dependencies count: {stats['crossing_dependencies_count']}")
    
    same_word_dists = stats['same_word_token_distances']
    if same_word_dists:
        print(f"Same-word token distances: {same_word_dists}")
        print(f"Average same-word distance: {statistics.mean(same_word_dists):.3f}")
        print(f"Max same-word distance: {max(same_word_dists)}")
    else:
        print("Same-word token distances: None (no linked tokens)")
    
    print()


def print_dependency_arcs(tokens: List[Dict[str, Any]], title: str = "Dependency Arcs"):
    """
    Print all dependency arcs for visualization and debugging.
    
    Args:
        tokens: List of token dictionaries
        title: Title for the output
    """
    print(f"\n{title}:")
    print("=" * 60)
    print(f"{'Arc':<20} {'Distance':<10} {'Relation':<12} {'Direction':<10}")
    print("-" * 60)
    
    for token in tokens:
        head_idx = token['head_index']
        dep_idx = token['index']
        distance = abs(head_idx - dep_idx)
        
        if head_idx == dep_idx:
            arc_str = f"ROOT({dep_idx})"
            direction = "self"
        else:
            head_token = tokens[head_idx]['text']
            dep_token = token['text']
            arc_str = f"{head_token}→{dep_token}"
            direction = "right" if head_idx < dep_idx else "left"
        
        print(f"{arc_str:<20} {distance:<10} {token['dep']:<12} {direction:<10}")
    
    print()

# Example usage:
if __name__ == "__main__":
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Example sentence
    sentence = "Timothy didn't boast about himself."
    original_doc = nlp(sentence)
    
    print("Original sentence:", repr(sentence))
    
    tokenizer = PERTURBATIONS["reverse_full"]["gpt2_tokenizer"]
    aligned_tokens = align_tokens_with_tokenizer(sentence, original_doc, tokenizer)
    
    print_aligned_tokens_simple(aligned_tokens, "Original Aligned Tokens")
    
    # Apply windowed shuffle perturbation
    shuffled_tokens = apply_windowed_shuffle_perturbation(aligned_tokens, window=3, seed=42)
    print_aligned_tokens_simple(shuffled_tokens, "After Windowed Shuffle (window=3, seed=42)")
    
    # Apply reverse perturbation
    reversed_tokens = apply_reverse_perturbation(aligned_tokens)
    print_aligned_tokens_simple(reversed_tokens, "After Reverse Perturbation")
    
    # Demonstrate that dependencies are preserved
    print("\nDependency Validation:")
    print("Original ROOT token:", [t['text'] for t in aligned_tokens if t['dep'] == 'ROOT'])
    print("Shuffled ROOT token:", [t['text'] for t in shuffled_tokens if t['dep'] == 'ROOT'])
    print("Reversed ROOT token:", [t['text'] for t in reversed_tokens if t['dep'] == 'ROOT'])

    print_dependency_statistics(aligned_tokens, "Original Dependency Statistics")
    print_dependency_arcs(aligned_tokens, "Original Dependency Arcs")
    print_dependency_statistics(shuffled_tokens, "Shuffled Dependency Statistics")
    print_dependency_arcs(shuffled_tokens, "Shuffled Dependency Arcs")
    print_dependency_statistics(reversed_tokens, "Reversed Dependency Statistics")
    print_dependency_arcs(reversed_tokens, "Reversed Dependency Arcs")
