import numpy as np
import torch
import sklearn
from pathlib import Path


def random_shuffle(o, random_state=None):
    res = sklearn.utils.shuffle(o, random_state=random_state)
    return res


def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a


def transfer_weights(
    model, weights_path: Path, device: torch.device = None, exclude_head: bool = True
):
    """Utility function that allows to easily transfer weights between models.
    Taken from the great self-supervised repository created by Kerem Turgutlu.
    https://github.com/KeremTurgutlu/self_supervised/blob/d87ebd9b4961c7da0efd6073c42782bbc61aaa2e/self_supervised/utils.py"""

    state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    matched_layers = 0
    unmatched_layers = []
    for name, param in state_dict.items():
        if exclude_head and "head" in name:
            continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f"check unmatched_layers: {unmatched_layers}")
        else:
            print(f"weights from {weights_path} successfully transferred!\n")


def rotate_axis0(o, steps=1):
    return o[np.arange(o.shape[0]) - steps]
