import random
import torch
import math

from torch.distributions.geometric import Geometric
from torch.distributions.binomial import Binomial


class SubsequenceMask:
    def __init__(
        self, r: float = 0.15, lm: int = 3, stateful: bool = True, sync: bool = False
    ):
        self.r = r
        self.lm = lm
        self.stateful = stateful
        self.sync = sync

    def __call__(self, o: torch.Tensor):
        device = o.device
        if o.ndim == 2:
            o = o[None]
        n_masks, mask_dims, mask_len = o.shape
        if self.sync == "random":
            self.sync = random.random() > 0.5
        dims = 1 if self.sync else mask_dims
        if self.stateful:
            numels = n_masks * dims * mask_len
            pm = torch.tensor([1 / self.lm], device=device)
            pu = torch.clip(pm * (self.r / max(1e-6, 1 - self.r)), 1e-3, 1)
            zot, proba_a, proba_b = (
                (torch.as_tensor([False, True], device=device), pu, pm)
                if random.random() > pm
                else (torch.as_tensor([True, False], device=device), pm, pu)
            )
            max_len = max(1, 2 * math.ceil(numels // (1 / pm + 1 / pu)))
            for i in range(10):
                _dist_a = (Geometric(probs=proba_a).sample([max_len]) + 1).long()
                _dist_b = (Geometric(probs=proba_b).sample([max_len]) + 1).long()
                dist_a = _dist_a if i == 0 else torch.cat((dist_a, _dist_a), dim=0)
                dist_b = _dist_b if i == 0 else torch.cat((dist_b, _dist_b), dim=0)
                add = torch.add(dist_a, dist_b)
                if torch.gt(torch.sum(add), numels):
                    break
            dist_len = torch.argmax((torch.cumsum(add, 0) >= numels).float()) + 1
            if dist_len % 2:
                dist_len += 1
            repeats = torch.cat((dist_a[:dist_len], dist_b[:dist_len]), -1).flatten()
            zot = zot.repeat(dist_len)
            mask = torch.repeat_interleave(zot, repeats)[:numels].reshape(
                n_masks, dims, mask_len
            )
        else:
            probs = torch.tensor(self.r, device=device)
            mask = Binomial(1, probs).sample((n_masks, dims, mask_len)).bool()
        if self.sync:
            mask = mask.repeat(1, mask_dims, 1)
        return mask
