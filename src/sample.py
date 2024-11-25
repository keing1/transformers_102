import torch as t
import torch.nn as nn
from typing import List, Optional

# Assuming no kv-caching

def sample(model: nn.Module, x: t.Tensor, max_tokens: int, temperature: float=1, p: Optional[float]=None, k: Optional[int]=None):
    # x: Tensor of integers of dim (b, s)
    assert temperature > 0
    if p is not None:
        assert p > 0 and p <= 1
    if k is not None:
        assert k >= 1
    
    for _ in range(max_tokens):
        pass

def beam_search_sampling(model: nn.Module, prompt: List[int], num_beams: int):
    assert num_beams > 0
    pass