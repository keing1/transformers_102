import unittest
import torch as t
import einops
from src import sample


class TestModelSampling(unittest.TestCase):
    def test_top_p(self):
        probs = t.tensor([[.4, .3, .3], [.5, .3, .2], [.1, .6, .3]])
        p = .75
        num_samples = 50000

        reference_probs = t.tensor([[.4,.3,.3],[.625,.375,0], [0, .6666666666, .3333333334]])

        total_tensor = t.zeros(probs.shape)

        for _ in range(num_samples):
            index_sample = sample.top_p_sample(probs, p)
            index_tensor = einops.repeat(t.arange(probs.shape[1]), 'p -> b p', b=probs.shape[0])
            increment_mask = index_tensor == index_sample
            total_tensor = total_tensor + increment_mask

        total_tensor = total_tensor / num_samples
        
        t.allclose(total_tensor, reference_probs, atol=3e-3, rtol=0)

    def test_top_k(self):
        probs = t.tensor([[.4, .2, .4], [.5, .3, .2], [.1, .4, .5]])
        k = 2
        num_samples = 50000

        reference_probs = t.tensor([[.5,0,.5],[.625,.375,0], [0, .444444444, .555555556]])

        total_tensor = t.zeros(probs.shape)

        for _ in range(num_samples):
            index_sample = sample.top_k_sample(probs, k)
            index_tensor = einops.repeat(t.arange(probs.shape[1]), 'p -> b p', b=probs.shape[0])
            increment_mask = index_tensor == index_sample
            total_tensor = total_tensor + increment_mask

        total_tensor = total_tensor / num_samples
        
        t.allclose(total_tensor, reference_probs, atol=3e-3, rtol=0)

if __name__ == '__main__':
    unittest.main()