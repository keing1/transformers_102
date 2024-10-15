import torch as t
from scipy.stats import norm

def relu(x: t.Tensor) -> t.Tensor:
    return t.maximum(x, t.zeros(x.shape))

def gelu(x: t.Tensor) -> t.Tensor:
    return x * 0.5 * (1 + t.erf(x / t.sqrt(t.tensor(2.0))))

def swish(x: t.Tensor, beta: float=1) -> t.Tensor:
    return x / (1 + t.exp(-beta * x))

def sigmoid(x: t.Tensor) -> t.Tensor:
    return 1/(1+t.exp(-x))