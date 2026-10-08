import torch
from torch import Tensor


def bounded_sigmoid(raw_opacity: Tensor, alpha_min: float) -> Tensor:
    if not 0.0 <= alpha_min < 1.0:
        raise ValueError(f"alpha_min must be in [0, 1), got {alpha_min}")
    return alpha_min + (1.0 - alpha_min) * torch.sigmoid(raw_opacity)


def bounded_logit(opacity: Tensor, alpha_min: float) -> Tensor:
    if not 0.0 <= alpha_min < 1.0:
        raise ValueError(f"alpha_min must be in [0, 1), got {alpha_min}")
    normalized = (opacity - alpha_min) / (1.0 - alpha_min)
    return torch.logit(normalized.clamp(1e-6, 1.0 - 1e-6))
