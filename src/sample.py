import torch as t
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def top_p_sample(probs: t.Tensor, p: float):
    sorted_probs, sorted_indices = t.sort(probs, descending=True)

    top_p_mask = t.cumsum(sorted_probs, dim=-1) - sorted_probs >= p

    masked_sorted_probs = t.masked_fill(sorted_probs, top_p_mask, 0)
    sampled_indices = t.multinomial(masked_sorted_probs, 1)
    original_indices = t.gather(sorted_indices, -1, sampled_indices)
    return original_indices


def top_k_sample(probs: t.Tensor, k: int):
    top_probs, top_indices = t.topk(probs, k)

    sampled_indices = t.multinomial(top_probs, 1)
    original_indices = t.gather(top_indices, -1, sampled_indices)
    return original_indices


def sample(model: nn.Module, x: t.Tensor, num_tokens: int, temperature: float=1, p: Optional[float]=None, k: Optional[int]=None):
    # x: Tensor of integers of dim (b, s)
    assert temperature > 0, "Temperature must be positive"
    if p is not None:
        assert p > 0 and p <= 1, "p must be between 0 and 1"
    if k is not None:
        assert k >= 1, "k must be at least 1"
    
    for _ in range(num_tokens):
        # Assuming no kv-caching
        logits = model(x)[:,-1]
        probs = F.softmax(logits/temperature, dim=-1)

        # Default to top p
        if p is not None:
            probs = top_p_sample(probs, p)
        elif k is not None:
            probs = top_k_sample(probs, k)
        
        x_next = t.multinomial(probs, 1, replacement=True)
        x = t.cat([x, x_next], dim=-1)    
    
    return x

def beam_search_sampling(model: nn.Module, prompt: List[int], num_beams: int):
    assert num_beams > 0
    pass