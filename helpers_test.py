import unittest
import torch
from helpers import create_mask, len_to_mask
from torch import Tensor

class TestHelpers(unittest.TestCase):

    def setUp(this):
        this.batch_size: int = 2
        this.sequence_length: int = 5
        this.max_len: int = 4
        this.max_length_seq: int = 5 
        this.hidden_size: int = 4
    
    def test_create_mask(this):

        x1: Tensor = torch.tensor([3, 4])
        x2: Tensor = torch.tensor([4, 5])
        expected: Tensor = torch.randn(this.batch_size, this.max_len, this.max_len)       
        out = create_mask(x1, x2, this.max_len, this.max_len)
        assert out.shape == expected.shape

        
    def test_len_to_mask(this):
        source: Tensor = torch.tensor([3, 2])
        out = len_to_mask(source, this.max_len)
        out_wo_len = len_to_mask(source)
        assert out.shape == (this.batch_size, this.max_len)
        assert out_wo_len.shape == (this.batch_size, source.max().item())

if __name__ == "__main__":
    unittest.main()
        