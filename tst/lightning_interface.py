from typing import Callable, Optional
import numpy as np
import torch
from torch.functional import Tensor
from torch.optim import Adam
import pytorch_lightning as pl
import torch.nn.functional as F

from tst.model.tst import TSTPlus


class LightningTST(pl.LightningModule):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        res_dropout: int = 0.0,
        act: str = "gelu",
        key_padding_mask: bool = True,
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        custom_head: bool = None,
        learning_rate=1e-3,
        masking_function: Callable[["torch.Tensor"], "torch.BoolTensor"] = None,
        verbose=False,
    ):
        if masking_function is None:
            raise RuntimeError(
                "Needs to pass a `masking_function`. \
                -> Function that takes the NN input as argument and return a boolean mask of same shape."
            )
        super().__init__()
        self.save_hyperparameters(ignore="verbose")
        self.model = TSTPlus(
            c_in=c_in,
            seq_len=seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            res_dropout=res_dropout,
            act=act,
            key_padding_mask=key_padding_mask,
            padding_var=padding_var,
            attn_mask=attn_mask,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            fc_dropout=fc_dropout,
            custom_head=custom_head,
            verbose=verbose,
        )
        self.masking_function = masking_function

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_train = batch["x"].to(dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mask = self.masking_function(x_train)
            x_masked = x_train.clone()
            x_masked[mask] = 0

        x_reconstructed, _ = self.model(x_train)
        # take the mask part of both x and reconstructed x
        loss = F.mse_loss(x_train[mask], x_reconstructed[mask])
        # compute loss and backprop on the masked part only
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_train = batch["x"].to(dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mask = self.masking_function(x_train)
            x_masked = x_train.clone()
            x_masked[mask] = 0

        x_reconstructed, _ = self.model(x_train)

        # take the mask part of both x and reconstructed x
        loss = F.mse_loss(x_train[mask], x_reconstructed[mask])
        # compute loss and backprop on the masked part only
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
