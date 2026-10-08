"""Conservative image marking candidates when pixel lane labels are absent."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def image_lane_candidates(rgb, road_mask, min_brightness=0.48, min_contrast=0.06,
                          window_size=15, white_chroma=0.22):
    if window_size < 3 or window_size % 2 != 1:
        raise ValueError("lane contrast window must be odd and >= 3")
    road = road_mask.bool()
    luminance = (rgb * rgb.new_tensor([0.2126, 0.7152, 0.0722])).sum(-1)
    # Normalize within road support: a dark non-road mask border is not evidence
    # of a marking. Large uniformly bright road regions are also not candidates.
    support = road.float()[None, None]
    pooling = dict(kernel_size=window_size, stride=1, padding=window_size//2)
    local = F.avg_pool2d((luminance * road)[None, None], **pooling)
    local = (local / F.avg_pool2d(support, **pooling).clamp_min(1e-6))[0, 0]
    white = rgb.amax(-1) - rgb.amin(-1) <= white_chroma
    yellow = (rgb[..., 0] >= 0.45) & (rgb[..., 1] >= 0.4) & (rgb[..., 2] < 0.75 * torch.minimum(rgb[..., 0], rgb[..., 1]))
    return road & (white | yellow) & (luminance >= min_brightness) & (luminance - local >= min_contrast)


def resize_lane_mask(mask, size):
    # Preserve thin markings during the initial low-resolution training stage.
    if tuple(mask.shape) == tuple(size):
        return mask.bool()
    source = mask.float()[None, None]
    if size[0] <= mask.shape[0] and size[1] <= mask.shape[1]:
        return F.adaptive_max_pool2d(source, size)[0, 0].bool()
    return F.interpolate(source, size=size, mode="nearest")[0, 0].bool()
