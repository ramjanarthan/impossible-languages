from typing import List
import torch

def get_perplexities(model, tokenizer, sentences: List[str], device="mps", max_length=512):
    """Calculate per-sentence perplexities"""

    # Process empty batch case
    if not sentences:
        return []

    with torch.no_grad():
        # Let the tokenizer handle padding and truncation directly
        inputs = tokenizer(
            sentences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )

        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        labels = input_ids.clone()

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

        # --- Vectorized Perplexity Calculation ---
        # Sum the loss for each sentence in the batch
        sentence_losses = loss.sum(dim=1)
        
        # Count the number of non-padded tokens in each sentence
        valid_token_counts = shift_attention_mask.sum(dim=1)

        # Avoid division by zero for empty or fully padded sentences
        # A count of 0 will result in a loss of 0, giving a perplexity of exp(0)=1. 
        # We can handle this by setting perplexity to infinity for such cases later.
        valid_token_counts[valid_token_counts == 0] = 1
        
        # Calculate the average loss for each sentence
        average_losses = sentence_losses / valid_token_counts
        
        # Compute perplexity
        perplexities = torch.exp(average_losses)
        
        # Manually set perplexity to infinity where there were no valid tokens
        perplexities[valid_token_counts == 0] = float('inf')

    return perplexities.tolist()