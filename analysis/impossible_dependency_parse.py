import spacy, json
from spacy import displacy
from typing import Dict, List, Tuple, Any

def create_perturbed_doc(original_sentence: str, original_doc, perturbed_sentence: str):
    """
    Create a manipulated Doc object that represents the perturbed sentence
    with the original dependency relationships preserved.
    
    Args:
        original_sentence: The original sentence string
        original_doc: The spaCy Doc object of the original sentence
        perturbed_sentence: The perturbed sentence string
        
    Returns:
        A pseudo-Doc object (dict) with remapped tokens and dependencies
    """
    
    # Step 1: Create character-level mapping
    char_mapping = map_characters(original_sentence, perturbed_sentence)
    
    # Step 2: Find where each original token appears in perturbed sentence
    token_positions = map_original_tokens_to_perturbed(original_doc, char_mapping, perturbed_sentence)
    
    # Step 3: Identify head tokens (leftmost fragments) for each original word
    head_tokens = identify_head_tokens(token_positions)
    
    # Step 4: Create perturbed token list
    perturbed_tokens = create_perturbed_tokens(original_doc, token_positions, head_tokens, perturbed_sentence)
    
    # Step 5: Remap dependencies
    dependency_mapping = remap_dependencies(original_doc, head_tokens, perturbed_tokens)
    
    # Step 6: Create pseudo-Doc object
    pseudo_doc = create_pseudo_doc(perturbed_tokens, dependency_mapping)
    
    return pseudo_doc


def map_characters(original: str, perturbed: str) -> Dict[int, int]:
    """
    Create a mapping from original character positions to perturbed positions.
    """
    # Remove spaces for mapping (since perturbations might change spacing)
    orig_chars = [c for c in original if c != ' ']
    pert_chars = [c for c in perturbed if c != ' ']
    
    # Simple character matching - assumes same characters, different order
    char_mapping = {}
    used_positions = set()
    
    for orig_idx, orig_char in enumerate(orig_chars):
        for pert_idx, pert_char in enumerate(pert_chars):
            if orig_char == pert_char and pert_idx not in used_positions:
                char_mapping[orig_idx] = pert_idx
                used_positions.add(pert_idx)
                break
    
    return char_mapping


def map_original_tokens_to_perturbed(original_doc, char_mapping: Dict[int, int], perturbed_sentence: str) -> Dict[int, List[Tuple[int, int, str]]]:
    """
    Map each original token to its fragments in the perturbed sentence.
    Returns: {original_token_idx: [(start_pos, end_pos, text), ...]}
    """
    token_positions = {}
    
    # Remove spaces to work with character positions
    orig_no_space = ''.join(c for c in original_doc.text if c != ' ')
    pert_no_space = ''.join(c for c in perturbed_sentence if c != ' ')
    
    for token in original_doc:
        if token.is_space or token.is_punct:
            continue
            
        token_positions[token.i] = []
        token_chars = [c for c in token.text if c != ' ']
        
        # Find where each character of this token ended up
        char_positions = []
        orig_char_start = sum(len([c for c in t.text if c != ' ']) for t in original_doc[:token.i])
        
        for i, char in enumerate(token_chars):
            orig_pos = orig_char_start + i
            if orig_pos in char_mapping:
                pert_pos = char_mapping[orig_pos]
                char_positions.append((pert_pos, char))
        
        # Group consecutive characters into fragments
        if char_positions:
            char_positions.sort()  # Sort by perturbed position
            
            current_start = char_positions[0][0]
            current_text = char_positions[0][1]
            
            for pos, char in char_positions[1:]:
                if pos == current_start + len(current_text):
                    # Consecutive character
                    current_text += char
                else:
                    # Gap found, save current fragment and start new one
                    token_positions[token.i].append((current_start, current_start + len(current_text), current_text))
                    current_start = pos
                    current_text = char
            
            # Add the last fragment
            token_positions[token.i].append((current_start, current_start + len(current_text), current_text))
    
    return token_positions


def identify_head_tokens(token_positions: Dict[int, List[Tuple[int, int, str]]]) -> Dict[int, Tuple[int, int, str]]:
    """
    Identify the head token (leftmost fragment) for each original token.
    """
    head_tokens = {}
    
    for orig_idx, fragments in token_positions.items():
        if fragments:
            # Find leftmost fragment (minimum start position)
            leftmost = min(fragments, key=lambda x: x[0])
            head_tokens[orig_idx] = leftmost
    
    return head_tokens


def create_perturbed_tokens(original_doc, token_positions: Dict, head_tokens: Dict, perturbed_sentence: str) -> List[Dict]:
    """
    Create a list of perturbed tokens with original linguistic information.
    """
    perturbed_tokens = []
    
    # Collect all head tokens with their positions
    positioned_heads = []
    for orig_idx, (start_pos, end_pos, text) in head_tokens.items():
        orig_token = original_doc[orig_idx]
        positioned_heads.append({
            'original_idx': orig_idx,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'text': text,
            'original_token': orig_token
        })
    
    # Sort by position in perturbed sentence
    positioned_heads.sort(key=lambda x: x['start_pos'])
    
    # Create new token list
    for new_idx, head_info in enumerate(positioned_heads):
        orig_token = head_info['original_token']
        perturbed_tokens.append({
            'i': new_idx,  # New position
            'original_i': head_info['original_idx'],  # Original position
            'text': head_info['text'],
            'lemma_': orig_token.lemma_,
            'pos_': orig_token.pos_,
            'dep_': orig_token.dep_,
            'is_punct': orig_token.is_punct,
            'is_space': orig_token.is_space,
            'start_pos': head_info['start_pos'],
            'end_pos': head_info['end_pos']
        })
    
    return perturbed_tokens


def remap_dependencies(original_doc, head_tokens: Dict, perturbed_tokens: List[Dict]) -> Dict[int, int]:
    """
    Remap dependency relationships based on new token positions.
    """
    # Create mapping from original index to new index
    orig_to_new = {}
    for new_idx, token_info in enumerate(perturbed_tokens):
        orig_to_new[token_info['original_i']] = new_idx
    
    dependency_mapping = {}
    
    for token_info in perturbed_tokens:
        orig_idx = token_info['original_i']
        new_idx = token_info['i']
        
        orig_token = original_doc[orig_idx]
        
        if orig_token.head == orig_token:
            # Root token
            dependency_mapping[new_idx] = new_idx
        else:
            # Find the head in the new arrangement
            orig_head_idx = orig_token.head.i
            if orig_head_idx in orig_to_new:
                new_head_idx = orig_to_new[orig_head_idx]
                dependency_mapping[new_idx] = new_head_idx
            else:
                # Fallback: make it root if head not found
                dependency_mapping[new_idx] = new_idx
    
    return dependency_mapping


def create_pseudo_doc(perturbed_tokens: List[Dict], dependency_mapping: Dict[int, int]):
    """
    Create a pseudo-Doc object that can be used with our analysis functions.
    """
    class PseudoToken:
        def __init__(self, token_info, head_idx, all_tokens):
            self.i = token_info['i']
            self.text = token_info['text']
            self.lemma_ = token_info['lemma_']
            self.pos_ = token_info['pos_']
            self.dep_ = token_info['dep_']
            self.is_punct = token_info['is_punct']
            self.is_space = token_info['is_space']
            self.idx = token_info['start_pos']  # Character position
            self._head_idx = head_idx
            self._all_tokens = all_tokens
            
        @property
        def head(self):
            if self._head_idx == self.i:
                return self  # Root
            return self._all_tokens[self._head_idx]
    
    class PseudoDoc:
        def __init__(self, tokens):
            self.tokens = tokens
            self.text = ''.join(token.text for token in tokens)
            
        def __iter__(self):
            return iter(self.tokens)
            
        def __getitem__(self, idx):
            if isinstance(idx, slice):
                return self.tokens[idx]
            return self.tokens[idx]
        
        def __len__(self):
            return len(self.tokens)
    
    # Create token objects
    tokens = []
    for token_info in perturbed_tokens:
        head_idx = dependency_mapping[token_info['i']]
        token = PseudoToken(token_info, head_idx, None)  # Will set all_tokens later
        tokens.append(token)
    
    # Set the all_tokens reference for each token
    for token in tokens:
        token._all_tokens = tokens
    
    return PseudoDoc(tokens)