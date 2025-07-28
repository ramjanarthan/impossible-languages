import spacy
from spacy.tokens import Doc
from transformers import GPT2Tokenizer
from typing import List, Dict, Tuple
from data_generation.utils.impossible_utils import PERTURBATIONS

# Load necessary models. In a real application, these might be passed as arguments.
# Using a blank model for vocab and a standard one for parsing.
nlp = spacy.load("en_core_web_sm")
gpt2_tokenizer = PERTURBATIONS["reverse_full"]["gpt2_tokenizer"]

def _align_spacy_to_gpt2(
    original_doc: Doc, original_gpt2_tokens: List[int], tokenizer: GPT2Tokenizer
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """
    Aligns spaCy tokens to GPT-2 tokens from the original sentence.

    This revised version is more robust and avoids searching for decoded substrings.
    It works by aligning character spans of both tokenization schemes.

    Args:
        original_doc: The spaCy Doc object for the original sentence.
        original_gpt2_tokens: The list of GPT-2 token IDs for the original sentence.
        tokenizer: The GPT-2 tokenizer instance.

    Returns:
        A tuple containing:
        - spacy_to_gpt2_map: A dictionary mapping a spaCy token index to a list of
          its corresponding GPT-2 token indices.
        - gpt2_to_spacy_map: A dictionary mapping a GPT-2 token index back to its
          corresponding spaCy token index.
    """
    spacy_to_gpt2_map = {i: [] for i in range(len(original_doc))}
    gpt2_to_spacy_map = {}
    original_text = original_doc.text

    # 1. Create a map from character index to its spaCy token index.
    char_to_spacy_index = [-1] * len(original_text)
    for token in original_doc:
        for i in range(token.idx, token.idx + len(token.text)):
            if i < len(char_to_spacy_index):
                char_to_spacy_index[i] = token.i

    # 2. Determine the character spans for each GPT-2 token.
    # This is tricky because decode() can be inconsistent. We find each token's
    # text in the original string sequentially.
    gpt2_token_texts = [tokenizer.decode([t]) for t in original_gpt2_tokens]
    char_cursor = 0
    for gpt2_idx, gpt2_text in enumerate(gpt2_token_texts):
        # Find the start of the gpt2 token text, searching from the last position
        try:
            # We use lstrip because gpt2 tokens often have a leading space 'Ġ'
            stripped_text = gpt2_text.lstrip()
            start_char = original_text.index(stripped_text, char_cursor)
            end_char = start_char + len(stripped_text)
            char_cursor = end_char

            # 3. Use the character maps to align.
            # Assign this gpt2 token to the spacy token that covers its first character.
            spacy_idx = char_to_spacy_index[start_char]
            if spacy_idx != -1:
                spacy_to_gpt2_map[spacy_idx].append(gpt2_idx)
                gpt2_to_spacy_map[gpt2_idx] = spacy_idx
        except ValueError:
            # Could not find the token text. This might happen with unusual tokens.
            # As a fallback, assign it to the last known spaCy token.
            if gpt2_idx > 0 and gpt2_idx - 1 in gpt2_to_spacy_map:
                spacy_idx = gpt2_to_spacy_map[gpt2_idx - 1]
                spacy_to_gpt2_map[spacy_idx].append(gpt2_idx)
                gpt2_to_spacy_map[gpt2_idx] = spacy_idx

    return spacy_to_gpt2_map, gpt2_to_spacy_map


def create_perturbed_doc(
    original_sentence: str,
    original_doc: Doc,
    tokenizer: GPT2Tokenizer,
    perturbed_gpt2_tokens: List[int],
) -> Doc:
    """
    Constructs a new spaCy Doc for a perturbed sentence, preserving the original
    dependency structure.

    Args:
        original_sentence: The original, unperturbed sentence string.
        original_doc: The result of nlp(original_sentence).
        tokenizer: The GPT-2 tokenizer used for the perturbation.
        perturbed_gpt2_tokens: The list of GPT-2 token IDs after perturbation.

    Returns:
        A new spaCy Doc object representing the parsed perturbed sentence.
    """
    # Step 1: Get original GPT-2 tokens and create alignment maps.
    original_gpt2_tokens = tokenizer.encode(original_sentence)
    spacy_to_gpt2_map, gpt2_to_spacy_map = _align_spacy_to_gpt2(
        original_doc, original_gpt2_tokens, tokenizer
    )

    # Step 2: Determine the new order of the original spaCy tokens.
    # We use the first gpt2 token of a group as its "anchor" to decide the group's position.
    reordered_spacy_indices = []
    placed_spacy_indices = set()

    # Handle cases where perturbation function might return tokens not in the original
    perturbed_gpt2_tokens_set = set(perturbed_gpt2_tokens)
    original_gpt2_tokens_set = set(original_gpt2_tokens)
    valid_perturbed_tokens = [t for t in perturbed_gpt2_tokens if t in original_gpt2_tokens_set]


    for gpt2_token_id in valid_perturbed_tokens:
        spacy_idx = gpt2_to_spacy_map.get(gpt2_token_id)
        if spacy_idx is not None and spacy_idx not in placed_spacy_indices:
            reordered_spacy_indices.append(spacy_idx)
            placed_spacy_indices.add(spacy_idx)
    
    # Ensure all original spacy tokens are placed, even if their gpt2 tokens were lost
    for i in range(len(original_doc)):
        if i not in placed_spacy_indices:
            reordered_spacy_indices.append(i)


    # Create the reordered list of original spaCy Token objects
    reordered_spacy_tokens = [original_doc[i] for i in reordered_spacy_indices]

    # Step 3: Generate the new words by decoding the reordered GPT-2 groups.
    new_words = []
    for spacy_idx in reordered_spacy_indices:
        gpt2_group = spacy_to_gpt2_map.get(spacy_idx, [])
        # Sort the gpt2 tokens in the group by their new position
        sorted_group = sorted(gpt2_group, key=lambda tid: perturbed_gpt2_tokens.index(tid) if tid in perturbed_gpt2_tokens_set else float('inf'))
        new_word = tokenizer.decode(sorted_group).strip()
        new_words.append(new_word if new_word else original_doc[spacy_idx].text) # Fallback for empty decodes

    # Step 4: Calculate the new head indices for the Doc constructor.
    # This requires mapping original tokens to their new positions.
    token_to_new_index_map = {token.i: i for i, token in enumerate(reordered_spacy_tokens)}

    new_heads_offsets = []
    for i, token in enumerate(reordered_spacy_tokens):
        # Find the original head of the token
        original_head = token.head
        
        if original_head.i == token.i: # It's the ROOT
            head_new_index = i
        else:
            # Find the new index of that head in the reordered list
            # The original head's original index is original_head.i
            head_new_index = token_to_new_index_map.get(original_head.i, i)

        # The 'heads' array for Doc expects an offset from the current token's index
        offset = head_new_index - i
        new_heads_offsets.append(offset)

    # Step 5: Assemble the new Doc object.
    # We also carry over POS, tags, and dependency labels from the original tokens.
    new_deps = [token.dep_ for token in reordered_spacy_tokens]
    new_pos = [token.pos_ for token in reordered_spacy_tokens]
    new_tags = [token.tag_ for token in reordered_spacy_tokens]

    # Ensure words list is not empty
    if not new_words:
        return Doc(original_doc.vocab)

    return Doc(
        original_doc.vocab,
        words=new_words,
        heads=new_heads_offsets,
        deps=new_deps,
        pos=new_pos,
        tags=new_tags,
    )

def print_doc_dependencies(doc: Doc):
    """Helper function to print dependencies in the desired format."""
    if not doc or len(doc) == 0:
        print("Document is empty.")
        return
    print(f"{'TEXT':<15} {'DEP':<10} {'HEAD':<15} {'HEAD_IDX'}")
    print("-" * 50)
    for token in doc:
        print(
            f"{token.text:<15} {token.dep_:<10} {token.head.text:<15} {token.head.i}"
        )

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Original Sentence and its "ground truth" parse
    sentence = "Timothy didn't boast about himself."
    original_doc = nlp(sentence)

    print("--- Original Dependency Parse ---")
    print_doc_dependencies(original_doc)
    print("\n" + "="*50 + "\n")

    # 2. Simulate a perturbation (full reverse)
    # This simulates the user's process: encode -> perturb -> get token list
    perturbed_gpt2_tokens = PERTURBATIONS["reverse_full"]["perturbation_function"](sentence)
    perturbed_sentence_decoded = gpt2_tokenizer.decode(perturbed_gpt2_tokens)

    print(f"Decoded Perturbed Sentence: '{perturbed_sentence_decoded}'")
    print("\n" + "="*50 + "\n")

    # 3. Create the new Doc object for the perturbed sentence
    perturbed_doc = create_perturbed_doc(
        original_sentence=sentence,
        original_doc=original_doc,
        tokenizer=gpt2_tokenizer,
        perturbed_gpt2_tokens=perturbed_gpt2_tokens,
    )

    print("--- Perturbed Dependency Parse (Projected) ---")
    print_doc_dependencies(perturbed_doc)

    # Example of how the new words are formed
    print("\n--- Explanation of New Word Formation ---")
    spacy_to_gpt2, _ = _align_spacy_to_gpt2(original_doc, original_gpt2_tokens, gpt2_tokenizer)
    print("Mapping from original spaCy token index to GPT-2 token indices:")
    print(spacy_to_gpt2)
    print("\nThe new words are formed by decoding the GPT-2 token groups in their new order.")

