import torch as t
import torch.nn.functional as F
import sys
import os

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def run_all_tests():
    test_activation_functions()
    print("All tests have passed!")

if __name__ == '__main__':
    run_all_tests()