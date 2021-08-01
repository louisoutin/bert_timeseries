import torch
from tst.masking import SubsequenceMask


def test_masking():
    BATCH_SIZE = 32
    DIM_INPUT = 5
    SEQ_lEN = 50

    input = torch.rand((BATCH_SIZE, DIM_INPUT, SEQ_lEN))

    masking_fct = SubsequenceMask()
    mask = masking_fct(input)
    assert mask.shape == input.shape
    assert mask.type() == "torch.BoolTensor"
