from src.model.attentions import T5Attention, T5LocalAttention, T5BlockAttention, T5TransientGlobalAttention
from src import T5Config
import torch

class TestAttentions:
    input_ = torch.rand(3, 100, 512).to('cuda')

    def test_full_attention_torch(self):
        config = T5Config(attention_type = 'full', use_triton=False)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_full_attention_triton(self):
        config = T5Config(attention_type = 'full', use_triton=True)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_local_attention_torch(self):
        config = T5Config(attention_type = 'local', use_triton=False)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_local_attention_triton(self):
        config = T5Config(attention_type = 'local', use_triton=True)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_block_attention_torch(self):
        config = T5Config(attention_type = 'block', use_triton=False)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_block_attention_triton(self):
        config = T5Config(attention_type = 'block', use_triton=True)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_transient_attention_torch(self):
        config = T5Config(attention_type = 'transient-global', use_triton=False)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)

    def test_transient_attention_triton(self):
        config = T5Config(attention_type = 'transient-global', use_triton=True)

        att_l = T5Attention(config).to('cuda')

        res = att_l(self.input_)
        assert res[0].shape == (3, 100, 512)