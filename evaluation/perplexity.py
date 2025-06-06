from scipy import stats
from typing import List
import torch
import math

def create_attention_mask(token_lists):
    seq_length = max([len(i) for i in token_lists])
    batch_size = len(token_lists)
    mask = torch.zeros((batch_size, seq_length))

    for i, tokens in enumerate(token_lists):
        mask[i, :len(tokens)] = 1

    return mask

def create_input_ids(token_lists, pad_token_id):
    seq_length = max([len(i) for i in token_lists])
    batch_size = len(token_lists)
    input_ids = torch.full((batch_size, seq_length), pad_token_id)

    for i, tokens in enumerate(token_lists):
        input_ids[i, :len(tokens)] = torch.tensor(tokens)

    return input_ids

# Calculate geometric mean 
def calculate_geometric_mean_perplexity(perplexities):
    """Calculate geometric mean of perplexities as mentioned in paper"""
    # Filter out any invalid perplexities
    valid_perps = [p for p in perplexities if p > 0 and not np.isnan(p) and not np.isinf(p)]
    if not valid_perps:
        return float('inf')

    # Geometric mean calculation
    return stats.gmean(valid_perps)


def get_perplexities(model, tokenizer, sentences: List[str], device="mps", max_length=512):
    """Calculate per-sentence perplexities"""

    # Process empty batch case
    if not sentences:
        return []

    with torch.no_grad():
        token_lists = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length).input_ids

        # Prepare data
        input_ids = create_input_ids(token_lists, tokenizer.pad_token_id).to(device)
        labels = input_ids.clone()  # GPT-2 uses input as labels for CLM task
        attention_mask = create_attention_mask(token_lists).to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        # The "shifted" nature of labels in GPT-2 (next token prediction)
        # Shift logits, labels, and attention mask by one position
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()

        # Instantiate loss function with no reduction
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

        # Calculate per-token loss
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Reshape back to the original batch size and sequence length
        loss = loss.view(shift_labels.size())

        # Apply the attention mask - only calculate loss where mask is 1
        loss = loss * shift_attention_mask

        # Sum the loss over the sequence length, get per-example perplexity
        per_example_loss = loss.sum(dim=1) / shift_attention_mask.sum(dim=1)
        return torch.exp(per_example_loss).tolist()