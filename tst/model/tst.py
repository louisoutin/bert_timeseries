import math
import numpy as np
import pytorch_lightning as pl

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from typing import Any, Callable, Optional
from collections import OrderedDict

import plotly as plt


from tst.model.pos_encoding import (
    Coord1dPosEncoding,
    Coord2dPosEncoding,
    PositionalEncoding,
)

from tst.model.layers import GACP1d, GAP1d, LinBnDrop, SigmoidRange
from tst.model.attention import MultiheadAttention


class _TSTEncoderLayer(nn.Module):
    def __init__(
        self,
        q_len: int,
        d_model: int,
        n_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        store_attn: bool = False,
        res_dropout: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        res_attention: bool = False,
        pre_norm: bool = False,
    ):
        super().__init__()
        assert (
            not d_model % n_heads
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = MultiheadAttention(
            d_model, n_heads, d_k, d_v, res_attention=res_attention
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(res_dropout)
        self.batchnorm_attn = nn.BatchNorm1d(q_len)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            self._get_activation_fn(activation),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        # Add & Norm
        self.dropout_ffn = nn.Dropout(res_dropout)
        self.batchnorm_ffn = nn.BatchNorm1d(q_len)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def forward(
        self,
        src: Tensor,
        prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.batchnorm_attn(src)  # Norm: batchnorm
        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )
        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.batchnorm_attn(src)  # Norm: batchnorm

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.batchnorm_ffn(src)  # Norm: batchnorm
        ## Position-wise Feed-Forward
        src2 = self.ff(src)
        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.batchnorm_ffn(src)  # Norm: batchnorm

        if self.res_attention:
            return src, scores
        else:
            return src

    def _get_activation_fn(self, activation):
        if callable(activation):
            return activation()
        elif activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        raise ValueError(
            f'{activation} is not available. You can use "relu", "gelu", or a callable'
        )


class _TSTEncoder(nn.Module):
    def __init__(
        self,
        q_len,
        d_model,
        n_heads,
        d_k=None,
        d_v=None,
        d_ff=None,
        res_dropout=0.0,
        activation="gelu",
        res_attention=False,
        n_layers=1,
        pre_norm: bool = False,
        store_attn: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _TSTEncoderLayer(
                    q_len,
                    d_model,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    d_ff=d_ff,
                    res_dropout=res_dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                )
                for i in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def forward(
        self,
        src: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )
            return output
        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            return output


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)

    def __repr__(self):
        if self.contiguous:
            return f"{self.__class__.__name__}(dims={', '.join([str(d) for d in self.dims])}).contiguous()"
        else:
            return (
                f"{self.__class__.__name__}({', '.join([str(d) for d in self.dims])})"
            )


class _TSTBackbone(nn.Module):
    def __init__(
        self,
        c_in: int,
        seq_len: int,
        max_seq_len: Optional[int] = 512,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        res_dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        key_padding_mask: bool = True,
        padding_var: Optional[int] = None,
        attn_mask: Optional[Tensor] = None,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        verbose: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__()
        # Input encoding
        q_len = seq_len
        self.new_q_len = False
        if (
            max_seq_len is not None and seq_len > max_seq_len
        ):  # Control temporal resolution
            self.new_q_len = True
            q_len = max_seq_len
            tr_factor = math.ceil(seq_len / q_len)
            total_padding = tr_factor * q_len - seq_len
            padding = (total_padding // 2, total_padding - total_padding // 2)
            self.W_P = nn.Sequential(
                nn.ConstantPad1d(padding, 0),
                nn.Conv1d(c_in, d_model, kernel_size=tr_factor, stride=tr_factor),
            )
            if verbose:
                print(
                    f"temporal resolution modified: {seq_len} --> {q_len} time steps: \
                        kernel_size={tr_factor}, stride={tr_factor}, padding={padding}.\n"
                )
        elif kwargs:
            self.new_q_len = True
            t = torch.rand(1, 1, seq_len)
            q_len = nn.Conv1d(1, 1, **kwargs)(t).shape[-1]
            self.W_P = nn.Conv1d(c_in, d_model, **kwargs)  # Eq 2
            if verbose:
                print(
                    f"Conv1d with kwargs={kwargs} applied to input to create input encodings\n"
                )
        else:
            self.W_P = nn.Linear(
                c_in, d_model
            )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        self.W_pos = self._positional_encoding(pe, learn_pe, q_len, d_model, device)

        # Residual dropout
        self.res_dropout = nn.Dropout(res_dropout)

        # Encoder
        self.encoder = _TSTEncoder(
            q_len,
            d_model,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            d_ff=d_ff,
            res_dropout=res_dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
        )
        self.transpose = Transpose(-1, -2, contiguous=True)
        self.key_padding_mask, self.padding_var, self.attn_mask = (
            key_padding_mask,
            padding_var,
            attn_mask,
        )

    def forward(self, x: Tensor) -> Tensor:  # x: [bs x nvars x q_len]

        # Padding mask
        if self.key_padding_mask:
            x, key_padding_mask = self._key_padding_mask(x)
        else:
            key_padding_mask = None

        # Input encoding
        if self.new_q_len:
            u = self.W_P(x).transpose(
                2, 1
            )  # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else:
            u = self.W_P(
                x.transpose(2, 1)
            )  # Eq 1        # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        u = self.res_dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(
            u, key_padding_mask=key_padding_mask, attn_mask=self.attn_mask
        )  # z: [bs x q_len x d_model]
        z = self.transpose(z)  # z: [bs x d_model x q_len]

        return z

    def _positional_encoding(self, pe, learn_pe, q_len, d_model, device):
        # Positional encoding
        if pe == None:
            W_pos = torch.zeros(
                (q_len, d_model), device=device
            )  # pe = None and learn_pe = False can be used to measure impact of pe
            learn_pe = False
        elif pe == "zero":
            W_pos = torch.zeros((q_len, 1), device=device)
        elif pe == "zeros":
            W_pos = torch.zeros((q_len, d_model), device=device)
        elif pe == "normal" or pe == "gauss":
            W_pos = torch.zeros((q_len, 1), device=device)
            torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
        elif pe == "uniform":
            W_pos = torch.zeros((q_len, 1), device=device)
            nn.init.uniform_(W_pos, a=0.0, b=0.1)
        elif pe == "lin1d":
            W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
        elif pe == "exp1d":
            W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
        elif pe == "lin2d":
            W_pos = Coord2dPosEncoding(
                q_len, d_model, exponential=False, normalize=True
            )
        elif pe == "exp2d":
            W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
        elif pe == "sincos":
            W_pos = PositionalEncoding(q_len, d_model, normalize=True)
        else:
            raise ValueError(
                f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
            'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
            )
        return nn.Parameter(W_pos, requires_grad=learn_pe)

    def _key_padding_mask(self, x):
        if self.padding_var is not None:
            mask = Tensor(x[:, self.padding_var] == 1)  # key_padding_mask: [bs x q_len]
            return x, mask
        else:
            mask = torch.isnan(x)
            x[mask] = 0
            if mask.any():
                mask = Tensor(
                    (mask.float().mean(1) == 1).bool()
                )  # key_padding_mask: [bs x q_len]
                return x, mask
            else:
                return x, None


class TSTPlus(nn.Module):
    """TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs"""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        max_seq_len: Optional[int] = 512,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_ff: int = 256,
        res_dropout: float = 0.0,
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
        custom_head: Optional[Callable] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues. Default=512.
            d_model: total dimension of the model (number of features created by the model). Default: 128 (range(64-512))
            n_heads:  parallel attention heads. Default:16 (range(8-16)).
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model. Default: 512 (range(256-512))
            res_dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            key_padding_mask: a boolean padding mask will be applied to attention if True to those steps in a sample where all features are nan.
            padding_var: (optional) an int indicating the variable that contains the padded steps (0: non-padded, 1: padded).
            attn_mask: a boolean mask will be applied to attention if a tensor of shape [min(seq_len, max_seq_len) x min(seq_len, max_seq_len)] if provided.
            res_attention: if True Residual MultiheadAttention is applied.
            pre_norm: if True normalization will be applied as the first step in the sublayers. Defaults to False
            store_attn: can be used to visualize attention weights. Default: False.
            num_layers: number of layers (or blocks) in the encoder. Default: 3 (range(1-4))
            pe: type of positional encoder.
                Available types (for experimenting): None, 'exp1d', 'lin1d', 'exp2d', 'lin2d', 'sincos', 'gauss' or 'normal',
                'uniform', 'zero', 'zeros' (default, as in the paper).
            learn_pe: learned positional encoder (True, default) or fixed positional encoder.
            flatten: this will flatten the encoder output to be able to apply an mlp type of head (default=False)
            fc_dropout: dropout applied to the final fully connected layer.
            concat_pool: indicates if global adaptive concat pooling will be used instead of global adaptive pooling.
            bn: indicates if batchnorm will be applied to the head.
            custom_head: custom head that will be applied to the network. It must contain all kwargs (pass a partial function)
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
        Input shape:
            x: bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
            attn_mask: q_len x q_len
            As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        """
        super().__init__()
        # Backbone
        self.backbone = _TSTBackbone(
            c_in,
            seq_len=seq_len,
            max_seq_len=max_seq_len,
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
            verbose=verbose,
            **kwargs,
        )

        # Head
        self.head_nf = d_model
        self.seq_len = self.backbone.seq_len
        if custom_head:
            self.head = custom_head(
                self.head_nf, c_in
            )  # custom head passed as a partial func with all its kwargs
        else:
            self.head = self.create_head(self.head_nf, c_in, fc_dropout=fc_dropout)

    def create_head(
        self,
        nf,
        c_in,
        fc_dropout=0.0,
    ):
        return nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Conv1d(nf, c_in, 1),
        )

    def forward(self, x):
        z = self.backbone(x)
        x_reconstructed = self.head(z)
        return x_reconstructed, z

    def show_pe(self, cmap="viridis", figsize=None):
        plt.figure(figsize=figsize)
        plt.pcolormesh(self.backbone.W_pos.detach().cpu().T, cmap=cmap)
        plt.title("Positional Encoding")
        plt.colorbar()
        plt.show()
        plt.figure(figsize=figsize)
        plt.title("Positional Encoding - value along time axis")
        plt.plot(F.relu(self.backbone.W_pos.data).mean(1).cpu())
        plt.plot(-F.relu(-self.backbone.W_pos.data).mean(1).cpu())
        plt.show()
