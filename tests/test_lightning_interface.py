import pytest
import torch
from tst.lightning_interface import LightningTST
from tst.masking import SubsequenceMask

BATCH_SIZE = 32
DIM_INPUT = 5
SEQ_lEN = 50
DIM_MODEL = 128


@pytest.fixture
def lightning_interface():
    masking_fct = SubsequenceMask()
    model = LightningTST(
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
        learning_rate=1e-3,
        masking_function=masking_fct,
    )
    return model


def test_forward_model(lightning_interface):
    x = torch.rand(
        BATCH_SIZE,  # batch size
        DIM_INPUT,
        SEQ_lEN,
    ).to("cpu")
    x_rec, z = lightning_interface(x)
    assert x_rec.shape == x.shape
    assert z.shape == (BATCH_SIZE, DIM_MODEL, SEQ_lEN)


def test_train_step(lightning_interface):
    batch = {
        "x": torch.rand(
            BATCH_SIZE,
            DIM_INPUT,
            SEQ_lEN,
        )
    }
    loss = lightning_interface.training_step(batch, batch_idx=0)
    assert loss.item() > 0


def test_val_step(lightning_interface):
    batch = {
        "x": torch.rand(
            BATCH_SIZE,
            DIM_INPUT,
            SEQ_lEN,
        )
    }
    loss = lightning_interface.validation_step(batch, batch_idx=0)
    assert loss.item() > 0
