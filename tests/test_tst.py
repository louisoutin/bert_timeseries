import torch
from tst.model.tst import TSTPlus


def test_tst():
    BATCH_SIZE = 32
    DIM_INPUT = 5
    SEQ_lEN = 50
    DIM_MODEL = 128

    model = TSTPlus(
        c_in=DIM_INPUT,
        seq_len=SEQ_lEN,
        n_layers=3,
        d_model=DIM_MODEL,
        n_heads=16,
        d_k=None,
        d_v=None,
        d_ff=256,
        res_dropout=0.0,
        act="gelu",
        key_padding_mask=True,
        padding_var=None,
        attn_mask=None,
        res_attention=True,
        pre_norm=False,
        store_attn=False,
        pe="zeros",
        learn_pe=True,
        fc_dropout=0.0,
        custom_head=None,
        verbose=False,
    )
    input = torch.rand((BATCH_SIZE, DIM_INPUT, SEQ_lEN))
    x_rec, z = model(input)
    assert x_rec.shape == input.shape
    assert z.shape == (BATCH_SIZE, DIM_MODEL, SEQ_lEN)
