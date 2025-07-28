import spacy
from typing import List, Dict, Any, Optional, Set
from data_generation.utils.impossible_utils import PERTURBATIONS

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
            'ent_type': ''
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


def print_aligned_tokens(tokens: List[Dict[str, Any]], title: str = "Aligned Tokens"):
    """
    Print aligned tokens in a nice tabular format.
    
    Args:
        tokens: List of token dictionaries from align_tokens_with_tokenizer
        title: Title for the output table
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
    Print aligned tokens in a simpler format, similar to the original example.
    
    Args:
        tokens: List of token dictionaries from align_tokens_with_tokenizer
        title: Title for the output table
    """
    print(f"\n{title}:")
    
    for token in tokens:
        head_token = tokens[token['head_index']]
        print(f"{token['index']:<3} {token['text']:<15} {token['dep']:<12} {head_token['text']:<15} {token['head_index']}")


# Example usage:
if __name__ == "__main__":
    import spacy
    
    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")
    
    # Example sentence
    sentence = "Timothy didn't boast about himself."
    original_doc = nlp(sentence)
    
    print("Original sentence:", repr(sentence))
    
    # Print original spaCy dependencies for comparison
    print("\nOriginal spaCy Dependencies:")
    print("=" * 60)
    for i, token in enumerate(original_doc):
        head_text = token.head.text if token.head.i != i else token.text
        print(f"{i:<3} {token.text:<15} {token.dep_:<12} {head_text:<15} {token.head.i}")
    
    tokenizer = PERTURBATIONS["reverse_full"]["gpt2_tokenizer"]
    aligned_doc = align_tokens_with_tokenizer(sentence, original_doc, tokenizer)
    
    # Print results
    print_aligned_tokens(aligned_doc, "Detailed Aligned Dependencies")
    print_aligned_tokens_simple(aligned_doc, "Simple Format - Aligned Dependencies with 'linked' relationships")

