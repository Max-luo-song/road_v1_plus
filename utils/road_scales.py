"""Road XY scale parameterization; Z remains a log standard deviation."""
import math

import torch


def encode_road_scales(scales, bounds):
    """Encode physical scales, retaining the legacy log encoding when disabled."""
    lower, upper = (float(v) for v in bounds)
    if lower < 0:
        return scales.log()
    if not 0 < lower < upper:
        raise ValueError("Road XY scale bounds must satisfy 0 < min < max")
    fraction = (scales[..., :2].log() - math.log(lower)) / math.log(upper / lower)
    xy = torch.logit(fraction.clamp(1e-6, 1 - 1e-6))
    return torch.cat((xy, scales[..., 2:].log()), dim=-1)


def decode_road_scales(raw, bounds):
    """Decode ordinary 3D RoadNodes scales, including legacy hard bounds."""
    lower, upper = (float(v) for v in bounds)
    if lower < 0:
        xy = raw[..., :2].exp().clamp(.02, .08)
    else:
        xy = (math.log(lower) + math.log(upper / lower) * raw[..., :2].sigmoid()).exp()
    z = raw[..., 2:].exp().clamp(.003, .01)
    return torch.cat((xy, z), dim=-1)
