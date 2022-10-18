import torch


def A_to_B_format(x: torch.Tensor):
    # batch_size, 1, sample_num
    FLU, FRD, BLD, BRU = x[:, 0:1, :], x[:, 1:2, :], x[:, 2:3, :], x[:, 3:, :]
    W = FLU + FRD + BLD + BRU
    X = FLU + FRD - BLD - BRU
    Y = FLU - FRD + BLD - BRU
    Z = FLU - FRD - BLD + BRU

    return torch.cat([W, X, Y, Z], dim=1)

def B_to_A_format(x: torch.Tensor):
    # batch_size, 1, sample_num
    W, X, Y, Z = x[:, 0:1, :], x[:, 1:2, :], x[:, 2:3, :], x[:, 3:, :]
    FLU = (W + X + Y + Z) / 4
    FRD = (W + X - Y - Z) / 4
    BLD = (W - X + Y - Z) / 4
    BRU = (W - X - Y + Z) / 4

    return torch.cat([FLU, FRD, BLD, BRU], dim=1)

def compute_radian(cos, sin):
    # use arccos will cause numerical unstable
    phi = torch.asin(sin)
    cond = torch.isclose(torch.cos(phi), cos, rtol=1e-5, atol=1e-5)   
    phi = torch.where(cond, phi, torch.pi-phi)
    phi = torch.where(phi <= torch.pi, phi, phi - 2 * torch.pi)
    return phi
