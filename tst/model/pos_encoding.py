import math
import torch


# Cell
def PositionalEncoding(q_len, d_model, normalize=True, device="cpu"):
    pe = torch.zeros(q_len, d_model, device=device)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe.to(device=device)


SinCosPosEncoding = PositionalEncoding

# Cell
def Coord2dPosEncoding(
    q_len,
    d_model,
    exponential=False,
    normalize=True,
    eps=1e-3,
    verbose=False,
    device="cpu",
):
    x = 0.5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        if verbose:
            print(f"{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}")
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe.to(device=device)


# Cell
def Coord1dPosEncoding(q_len, exponential=False, normalize=True, device="cpu"):
    cpe = (
        2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
        - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe.to(device=device)
