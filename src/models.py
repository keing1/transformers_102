import torch as t

from src import layers, blocks

class GPT2Model(t.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        pass