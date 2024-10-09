import torch as t
import torch.nn.functional as F
import sys
import os

from src import activations

def test_activation_functions():
    x = t.randn(100)
    x2 = x.reshape((10,10))
    assert t.allclose(activations.relu(x), F.relu(x))
    assert t.allclose(activations.relu(x2), F.relu(x2))

    assert t.allclose(activations.gelu(x), F.gelu(x))
    assert t.allclose(activations.gelu(x2), F.gelu(x2))

    assert t.allclose(activations.swish(x), F.silu(x))
    assert t.allclose(activations.swish(x2), F.silu(x2))

def test_normalizers():
    pass

def test_embedding():
    pass

def test_linear():
    pass

def test_attention():
    pass

def test_mlp_blocks():
    pass

def run_all_tests():
    test_activation_functions()
    test_normalizers()
    test_embedding()
    test_linear()
    print("All tests have passed!")

if __name__ == '__main__':
    run_all_tests()