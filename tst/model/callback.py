# import torch
# import math
# import random
# import warnings
# import os
# import numpy as np
# import pytorch_lightning as pl

# from typing import Any, Callable, Optional
# from pathlib import Path

# from torch import nn
# from torch.distributions.beta import Beta
# from torch.distributions.geometric import Geometric
# from torch.distributions.binomial import Binomial

# from pytorch_lightning.callbacks import Callback

# import matplotlib.colors as mcolors
# import plotly as plt

# from tst.model.utils import rotate_axis0, transfer_weights, pv, random_shuffle


# def create_subsequence_mask(o, r=0.15, lm=3, stateful=True, sync=False):
#     device = o.device
#     if o.ndim == 2:
#         o = o[None]
#     n_masks, mask_dims, mask_len = o.shape
#     if sync == "random":
#         sync = random.random() > 0.5
#     dims = 1 if sync else mask_dims
#     if stateful:
#         numels = n_masks * dims * mask_len
#         pm = torch.tensor([1 / lm], device=device)
#         pu = torch.clip(pm * (r / max(1e-6, 1 - r)), 1e-3, 1)
#         zot, proba_a, proba_b = (
#             (torch.as_tensor([False, True], device=device), pu, pm)
#             if random.random() > pm
#             else (torch.as_tensor([True, False], device=device), pm, pu)
#         )
#         max_len = max(1, 2 * math.ceil(numels // (1 / pm + 1 / pu)))
#         for i in range(10):
#             _dist_a = (Geometric(probs=proba_a).sample([max_len]) + 1).long()
#             _dist_b = (Geometric(probs=proba_b).sample([max_len]) + 1).long()
#             dist_a = _dist_a if i == 0 else torch.cat((dist_a, _dist_a), dim=0)
#             dist_b = _dist_b if i == 0 else torch.cat((dist_b, _dist_b), dim=0)
#             add = torch.add(dist_a, dist_b)
#             if torch.gt(torch.sum(add), numels):
#                 break
#         dist_len = torch.argmax((torch.cumsum(add, 0) >= numels).float()) + 1
#         if dist_len % 2:
#             dist_len += 1
#         repeats = torch.cat((dist_a[:dist_len], dist_b[:dist_len]), -1).flatten()
#         zot = zot.repeat(dist_len)
#         mask = torch.repeat_interleave(zot, repeats)[:numels].reshape(
#             n_masks, dims, mask_len
#         )
#     else:
#         probs = torch.tensor(r, device=device)
#         mask = Binomial(1, probs).sample((n_masks, dims, mask_len)).bool()
#     if sync:
#         mask = mask.repeat(1, mask_dims, 1)
#     return mask


# def create_variable_mask(o, r=0.15):
#     device = o.device
#     n_masks, mask_dims, mask_len = o.shape
#     _mask = torch.zeros((n_masks * mask_dims, mask_len), device=device)
#     if int(mask_dims * r) > 0:
#         n_masked_vars = int(n_masks * mask_dims * r)
#         p = torch.tensor([1.0 / (n_masks * mask_dims)], device=device).repeat(
#             [n_masks * mask_dims]
#         )
#         sel_dims = p.multinomial(num_samples=n_masked_vars, replacement=False)
#         _mask[sel_dims] = 1
#     mask = _mask.reshape(*o.shape).bool()
#     return mask


# def create_future_mask(o, r=0.15, sync=False):
#     if o.ndim == 2:
#         o = o[None]
#     n_masks, mask_dims, mask_len = o.shape
#     if sync == "random":
#         sync = random.random() > 0.5
#     dims = 1 if sync else mask_dims
#     probs = torch.tensor(r, device=o.device)
#     mask = Binomial(1, probs).sample((n_masks, dims, mask_len))
#     if sync:
#         mask = mask.repeat(1, mask_dims, 1)
#     mask = torch.sort(mask, dim=-1, descending=True)[0].bool()
#     return mask


# def natural_mask(o):
#     """Applies natural missingness in a batch to non-nan values in the next sample"""
#     mask1 = torch.isnan(o)
#     mask2 = rotate_axis0(mask1)
#     return torch.logical_and(mask2, ~mask1)


# # Cell
# def create_mask(
#     o,
#     r=0.15,
#     lm=3,
#     stateful=True,
#     sync=False,
#     subsequence_mask=True,
#     variable_mask=False,
#     future_mask=False,
# ):
#     if r <= 0 or r >= 1:
#         return torch.ones_like(o)
#     if int(r * o.shape[1]) == 0:
#         variable_mask = False
#     if subsequence_mask and variable_mask:
#         random_thr = 1 / 3 if sync == "random" else 1 / 2
#         if random.random() > random_thr:
#             variable_mask = False
#         else:
#             subsequence_mask = False
#     elif future_mask:
#         return create_future_mask(o, r=r)
#     elif subsequence_mask:
#         return create_subsequence_mask(o, r=r, lm=lm, stateful=stateful, sync=sync)
#     elif variable_mask:
#         return create_variable_mask(o, r=r)
#     else:
#         raise ValueError(
#             "You need to set subsequence_mask, variable_mask or future_mask to True or pass a custom mask."
#         )


# class MVP(Callback):
#     order = 60

#     def __init__(
#         self,
#         r: float = 0.15,
#         subsequence_mask: bool = True,
#         lm: float = 3.0,
#         stateful: bool = True,
#         sync: bool = False,
#         variable_mask: bool = False,
#         future_mask: bool = False,
#         custom_mask: Optional[Callable] = None,
#         nan_to_num: int = 0,
#         dropout: float = 0.1,
#         crit: callable = None,
#         weights_path: Optional[str] = None,
#         target_dir: str = "./data/MVP",
#         fname: str = "model",
#         save_best: bool = True,
#         verbose: bool = False,
#     ):
#         r"""
#         Callback used to perform the pretext task of reconstruct the original data after a binary mask has been applied.

#         Args:
#             r: proba of masking.
#             subsequence_mask: apply a mask to random subsequences.
#             lm: average mask len when using stateful (geometric) masking.
#             stateful: geometric distribution is applied so that average mask length is lm.
#             sync: all variables have the same masking.
#             variable_mask: apply a mask to random variables. Only applicable to multivariate time series.
#             future_mask: used to train a forecasting model.
#             custom_mask: allows to pass any type of mask with input tensor and output tensor. Values to mask should be set to True.
#             nan_to_num: integer used to fill masked values
#             dropout: dropout applied to the head of the model during pretraining.
#             crit: loss function that will be used. If None MSELossFlat().
#             weights_path: indicates the path to pretrained weights. This is useful when you want to continue training from a checkpoint. It will load the
#                           pretrained weights to the model with the MVP head.
#             target_dir : directory where trained model will be stored.
#             fname : file name that will be used to save the pretrained model.
#             save_best: saves best model weights
#         """
#         assert (
#             subsequence_mask or variable_mask or future_mask or custom_mask
#         ), "you must set (subsequence_mask and/or variable_mask) or future_mask to True or use a custom_mask"
#         if custom_mask is not None and (
#             future_mask or subsequence_mask or variable_mask
#         ):
#             warnings.warn("Only custom_mask will be used")
#         elif future_mask and (subsequence_mask or variable_mask):
#             warnings.warn("Only future_mask will be used")
#         self.subsequence_mask
#         self.variable_mask
#         self.future_mask
#         self.custom_mask
#         self.dropout
#         self.r = r
#         self.lm = lm
#         self.stateful = stateful
#         self.sync = sync
#         self.crit = crit
#         self.weights_path = weights_path
#         self.fname = fname
#         self.save_best = save_best
#         self.verbose = verbose
#         self.nan_to_num = nan_to_num
#         self.PATH = Path(f"{target_dir}/{self.fname}")
#         if not os.path.exists(self.PATH.parent):
#             os.makedirs(self.PATH.parent)
#         self.path_text = f"pretrained weights_path='{self.PATH}.pth'"

#     def on_train_batch_start(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         batch: Any,
#         batch_idx: int,
#         dataloader_idx: int,
#     ) -> None:
#         """Called when the train batch begins."""
#         x = batch["x"]
#         original_mask = torch.isnan(x)

#         if self.custom_mask is not None:
#             new_mask = self.custom_mask(self.x)
#         else:
#             new_mask = create_mask(
#                 x,
#                 r=self.r,
#                 lm=self.lm,
#                 stateful=self.stateful,
#                 sync=self.sync,
#                 subsequence_mask=self.subsequence_mask,
#                 variable_mask=self.variable_mask,
#                 future_mask=self.future_mask,
#             ).bool()
#         if original_mask.any():
#             self.mask = torch.logical_and(new_mask, ~original_mask)
#         else:
#             self.mask = new_mask
#         self.learn.yb = (torch.nan_to_num(self.x, self.nan_to_num),)
#         self.learn.xb = (self.yb[0].masked_fill(self.mask, self.nan_to_num),)

#     def after_epoch(self):
#         val = self.learn.recorder.values[-1][-1]
#         if self.save_best:
#             if np.less(val, self.best):
#                 self.best = val
#                 self.best_epoch = self.epoch
#                 torch.save(self.learn.model.state_dict(), f"{self.PATH}.pth")
#                 pv(
#                     f"best epoch: {self.best_epoch:3}  val_loss: {self.best:8.6f} - {self.path_text}",
#                     self.verbose or (self.epoch == self.n_epoch - 1),
#                 )
#             elif self.epoch == self.n_epoch - 1:
#                 print(
#                     f"\nepochs: {self.n_epoch} best epoch: {self.best_epoch:3}  val_loss: {self.best:8.6f} - {self.path_text}\n"
#                 )

#     def after_fit(self):
#         self.run = True

#     def _loss(self, preds, target):
#         return self.crit(preds[self.mask], target[self.mask])

#     def show_preds(
#         self, max_n=9, nrows=3, ncols=3, figsize=None, sharex=True, **kwargs
#     ):
#         b = self.learn.dls.valid.one_batch()
#         self.learn._split(b)
#         xb = self.xb[0].detach().cpu().numpy()
#         bs, nvars, seq_len = xb.shape
#         self.learn("before_batch")
#         masked_pred = (
#             torch.where(
#                 self.mask,
#                 self.learn.model(*self.learn.xb),
#                 torch.tensor([np.nan], device=self.learn.x.device),
#             )
#             .detach()
#             .cpu()
#             .numpy()
#         )
#         ncols = min(ncols, math.ceil(bs / ncols))
#         nrows = min(nrows, math.ceil(bs / ncols))
#         max_n = min(max_n, bs, nrows * ncols)
#         if figsize is None:
#             figsize = (ncols * 6, math.ceil(max_n / ncols) * 4)
#         fig, ax = plt.subplots(
#             nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, **kwargs
#         )
#         idxs = np.random.permutation(np.arange(bs))
#         colors = list(mcolors.TABLEAU_COLORS.keys()) + random_shuffle(
#             list(mcolors.CSS4_COLORS.keys())
#         )
#         i = 0
#         for row in ax:
#             for col in row:
#                 color_iter = iter(colors)
#                 for j in range(nvars):
#                     try:
#                         color = next(color_iter)
#                     except:
#                         color_iter = iter(colors)
#                         color = next(color_iter)
#                     col.plot(xb[idxs[i]][j], alpha=0.5, color=color)
#                     col.plot(
#                         masked_pred[idxs[i]][j],
#                         marker="o",
#                         markersize=4,
#                         linestyle="None",
#                         color=color,
#                     )
#                 i += 1
#         plt.tight_layout()
#         plt.show()


# TSBERT = MVP
