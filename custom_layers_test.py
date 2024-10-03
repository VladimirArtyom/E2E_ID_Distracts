import unittest
import torch
from torch import Tensor
import random
import numpy as np

from custom_layers import DistractorEncoder
from backup_2 import DGEncoder
from fairseq.data import Dictionary

def define_dictionary() -> Dictionary:
    d = Dictionary()
    d.add_symbol("<unk>")
    d.add_symbol("<pad>")
    d.add_symbol("<sep>")
    d.add_symbol("the")
    d.add_symbol("quick")
    d.add_symbol("brown")
    d.add_symbol("jump")
    d.add_symbol("baas")
    return d

class TestDistractorEncoder(unittest.TestCase):
    def setUp(this):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        dictionary = define_dictionary()
        this.embed_dim=300
        this.hidden_size=300
        this.batch_size = 5
        this.num_layers=1
        this.dropout_out=0
        this.dropout_in=0
        this.bidirectional=False
        this.left_pad=False
        this.pretrained_embed=None
        this.padding_value=0
        this.seq_length_max = 5
        
        this.encoder = DGEncoder(dictionary,
                                         embed_dim=this.embed_dim,
                                         hidden_size=this.hidden_size,
                                         num_layers=this.num_layers,
                                         dropout_out=this.dropout_out,
                                         dropout_in=this.dropout_in,
                                         bidirectional=this.bidirectional,
                                         left_pad=this.left_pad,
                                         pretrained_embed=this.pretrained_embed,
                                         padding_value=this.padding_value
                                         )
        this.cEncoder = DistractorEncoder(dictionary,
                                        embed_dim=this.embed_dim,
                                         hidden_size=this.hidden_size,
                                         num_layers=this.num_layers,
                                         dropout_out=this.dropout_out,
                                         dropout_in=this.dropout_in,
                                         bidirectional=this.bidirectional,
                                         left_pad=this.left_pad,
                                         pretrained_embed=this.pretrained_embed,
                                         padding_value=this.padding_value)
    @unittest.skip(reason="")
    def test_encode_text(this):
        tokens = torch.tensor([
            [1, 2, 3, 0, 0],
            [4, 5, 6, 7, 9],
            [9, 3, 0, 0, 0],
            [4, 5, 1, 9, 0],
        ], dtype=torch.long)

        lengths = torch.tensor([3, 5, 2, 4], dtype=torch.long)
        x, hds, cds = this.encoder.encode_text(tokens, lengths, required_embed=True)
        seq_length_max = this.seq_length_max
        hidden_size = this.hidden_size
        batch_size = this.batch_size
        num_layer= this.num_layers
        assert x.shape == torch.Size([batch_size, seq_length_max, hidden_size])
        assert hds.shape == torch.Size([num_layer, batch_size, hidden_size])
        assert cds.shape == torch.Size([num_layer, batch_size, hidden_size])

    @unittest.skip(reason="")
    def test_gate_self_attention(this):
        output_units: int = this.hidden_size
        batch_size: int = 2
        sequence_length: int = 5
        sequences: Tensor = torch.randn(batch_size, sequence_length, output_units)
        expected = torch.tensor([0.3474, 0.5422], dtype=torch.float)
        #mask_example : # Plus tard
        res = this.encoder.gate_self_attention(sequences)
        torch.equal( res, expected)

    @unittest.skip(reason="")
    def test_combine_bidirectional(this):
        batch_size: int = 2
        num_layers: int = 1
        hidden_size: int = 4
        outputs: Tensor = torch.randn(num_layers, batch_size, hidden_size)
        expected: Tensor = torch.randn(num_layers, batch_size, hidden_size)

        res = this.encoder.combine_bidirectional(outputs, batch_size)
        assert res.shape == expected.shape
        
    def test_forward(this):
        source_tokens: Tensor = torch.tensor([ [1, 2, 3, 0, 0],
                                               [1, 4, 0, 0, 0],
                                               [2, 2, 1, 0, 0],
                                               [3, 4, 2, 1, 4],
                                               [4, 4, 5, 2, 1]])
        question_tokens: Tensor = torch.tensor([ [6, 2, 3, 0, 0],
                                               [4, 4, 0, 0, 0],
                                               [3, 2, 1, 1, 0],
                                               [3, 4, 2, 1, 4],
                                               [4, 4, 5, 2, 1]])
        answer_tokens: Tensor = torch.tensor([ [2, 2, 3, 0, 0],
                                               [6, 4, 0, 0, 0],
                                               [5, 2, 1, 0, 0],
                                               [5, 5, 2, 1, 4],
                                               [2, 1, 5, 2, 1]])

        source_lengths: Tensor = torch.tensor([3, 2, 3, 5, 5])
        question_lengths: Tensor = torch.tensor([3, 2, 4, 5, 5])
        answer_lengths: Tensor = torch.tensor([3, 2, 3, 5, 5])
        this.encoder.eval()
        this.cEncoder.eval()
        q,a = this.encoder.forward(source_tokens, source_lengths,
                              question_tokens, question_lengths,
                                answer_tokens, answer_lengths)
        qc, ac = this.cEncoder.forward(source_tokens, source_lengths,
                                        question_tokens, question_lengths,
                                          answer_tokens, answer_lengths)
        #assert q.shape ==  qc.shape
        #assert a.shape == ac.shape
if __name__ == "__main__":
    unittest.main()