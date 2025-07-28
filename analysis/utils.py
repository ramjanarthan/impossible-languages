from typing import List, Dict

def create_spacy_perturbation_map(spacy_doc, gpt2_tokenizer, perturbation_type: str) -> List[int]:
    """
    Creates a perturbation map for spaCy tokens based on a specified perturbation
    of the underlying GPT-2 tokens.

    Args:
        spacy_doc: The spaCy Doc of the original sentence.
        gpt2_tokenizer: The GPT-2 tokenizer used for perturbation.
        perturbation_type: The type of perturbation to apply (e.g., 'reverse_full').

    Returns:
        A list where `map[new_position] = old_position` for spaCy tokens.
    """
    # 1. Align spaCy tokens to GPT-2 sub-tokens
    encoding = gpt2_tokenizer(spacy_doc.text, return_offsets_mapping=True)
    gpt2_offsets = encoding['offset_mapping']
    
    spacy_to_gpt2_indices = {i: [] for i in range(len(spacy_doc))}
    gpt2_to_spacy_index = {}

    for gpt2_idx, (start, end) in enumerate(gpt2_offsets):
        if start == end: continue
        for spacy_token in spacy_doc:
            if start >= spacy_token.idx and end <= (spacy_token.idx + len(spacy_token.text_with_ws)):
                spacy_to_gpt2_indices[spacy_token.i].append(gpt2_idx)
                gpt2_to_spacy_index[gpt2_idx] = spacy_token.i
                break

    num_gpt2_tokens = len(gpt2_offsets)
    original_gpt2_order = list(range(num_gpt2_tokens))

    # 2. Perturb the list of GPT-2 token indices
    if perturbation_type == 'reverse_full':
        perturbed_gpt2_order = original_gpt2_order[::-1]
    else:
        # Placeholder for other perturbation types
        perturbed_gpt2_order = original_gpt2_order

    # Create a map from the old GPT-2 index to its new position
    gpt2_old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(perturbed_gpt2_order)}

    # 3. Determine the new position for each spaCy token
    spacy_new_positions = {}
    for spacy_idx, gpt2_indices in spacy_to_gpt2_indices.items():
        if not gpt2_indices:
            continue
        # The spaCy token's new position is determined by its first GPT-2 sub-token.
        first_gpt2_idx = gpt2_indices[0]
        new_pos = gpt2_old_to_new_map[first_gpt2_idx]
        spacy_new_positions[spacy_idx] = new_pos

    # 4. Create the final perturbation map for spaCy tokens.
    # Sort by the new position to get the final order of original spaCy tokens.
    sorted_by_new_pos = sorted(spacy_new_positions.items(), key=lambda item: item[1])
    
    final_map = [-1] * len(spacy_doc)
    final_spacy_order = [item[0] for item in sorted_by_new_pos]

    # Ensure all spacy tokens are included, even if they didn't map to a gpt2 token
    unmapped_spacy = [i for i in range(len(spacy_doc)) if i not in final_spacy_order]
    full_spacy_order = final_spacy_order + unmapped_spacy

    # The final map should be [new_pos] -> old_pos
    # We have the order of old_pos, so we invert it.
    for new_idx, old_idx in enumerate(full_spacy_order):
        if new_idx < len(final_map):
            final_map[new_idx] = old_idx

    return final_map
