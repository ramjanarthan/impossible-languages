from typing import List


class PseudoToken:
    """A lightweight, pickleable substitute for a spaCy Token."""
    def __init__(self, token, new_i, new_head_i):
        self.i = new_i
        self.text = token.text
        self.lemma_ = token.lemma_
        self.pos_ = token.pos_
        self.dep_ = token.dep_
        self.is_punct = token.is_punct
        self.is_space = token.is_space
        self.idx = token.idx # Keep original char offset for now

        self._new_head_i = new_head_i
        self._doc = None # This will be a PseudoDoc

    @property
    def head(self):
        """Return the head token from the pseudo-doc."""
        return self._doc[self._new_head_i]


class PseudoDoc:
    """A lightweight, pickleable substitute for a spaCy Doc."""
    def __init__(self, tokens):
        self.tokens = tokens
        self.text = " ".join(t.text for t in tokens)
        for token in self.tokens:
            token._doc = self

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item]


def create_perturbed_doc(original_doc, perturbation_map: List[int]):
    """
    Creates a reordered pseudo-Doc from an original Doc and a perturbation map.

    Args:
        original_doc: The spaCy Doc object of the original sentence.
        perturbation_map: A list where `perturbation_map[new_position] = old_position`.

    Returns:
        A PseudoDoc object with tokens in the new order but preserving
        original dependency relationships.
    """
    original_tokens = list(original_doc)
    reordered_tokens = [original_tokens[i] for i in perturbation_map if i != -1]

    # Mapping from old index to new index for head calculations
    old_to_new_idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(perturbation_map) if old_idx != -1}

    pseudo_tokens = []
    for new_i, token in enumerate(reordered_tokens):
        # Find the new index of this token's original head
        original_head_i = token.head.i
        new_head_i = old_to_new_idx_map.get(original_head_i, new_i) # Default to self if head is not in map

        # If the token is its own head (root), it should point to itself in the new order
        if token.i == original_head_i:
            new_head_i = new_i

        pt = PseudoToken(token, new_i=new_i, new_head_i=new_head_i)
        pseudo_tokens.append(pt)

    return PseudoDoc(pseudo_tokens)