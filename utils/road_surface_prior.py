import math
from typing import Tuple

import torch


@torch.no_grad()
def build_road_surface_grid(
    road_points: torch.Tensor,
    cell_size: float,
    max_grid_cells: int = 4_000_000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Aggregate road heights into a bounded XY grid."""
    if road_points.ndim != 2 or road_points.shape[1] != 3 or not road_points.numel():
        raise ValueError("road_points must be a non-empty [N, 3] tensor")
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")

    xy_min = road_points[:, :2].amin(dim=0)
    xy_max = road_points[:, :2].amax(dim=0)
    span = (xy_max - xy_min).clamp_min(0)
    effective_cell_size = float(cell_size)

    while True:
        shape = torch.ceil(span / effective_cell_size).long() + 1
        width, height = int(shape[0].item()), int(shape[1].item())
        if width * height <= max_grid_cells:
            break
        effective_cell_size *= max(
            1.01, math.sqrt((width * height) / max_grid_cells)
        )

    indices = torch.floor(
        (road_points[:, :2] - xy_min) / effective_cell_size
    ).long()
    indices[:, 0].clamp_(0, width - 1)
    indices[:, 1].clamp_(0, height - 1)
    flat_indices = indices[:, 1] * width + indices[:, 0]

    height_sums = torch.zeros(
        height * width, dtype=road_points.dtype, device=road_points.device
    )
    counts = torch.zeros_like(height_sums)
    height_sums.scatter_add_(0, flat_indices, road_points[:, 2])
    counts.scatter_add_(0, flat_indices, torch.ones_like(road_points[:, 2]))
    valid = counts > 0
    heights = height_sums / counts.clamp_min(1)

    return (
        xy_min,
        heights.view(height, width),
        valid.view(height, width),
        effective_cell_size,
    )


def road_intrusion_confidence(
    means: torch.Tensor,
    grid_origin: torch.Tensor,
    grid_heights: torch.Tensor,
    grid_valid: torch.Tensor,
    cell_size: float,
    below_margin: float,
    above_margin: float,
    transition: float,
) -> torch.Tensor:
    """Return soft confidence that Gaussian centers occupy the road surface slab."""
    if not means.numel():
        return means.new_zeros((0,))

    indices = torch.floor((means[:, :2] - grid_origin) / cell_size).long()
    height, width = grid_heights.shape
    inside = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < width)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < height)
    )
    x = indices[:, 0].clamp(0, width - 1)
    y = indices[:, 1].clamp(0, height - 1)
    supported = inside & grid_valid[y, x]

    road_height = grid_heights[y, x]
    height_delta = means[:, 2] - road_height
    transition = max(float(transition), 1e-6)
    above_lower_bound = torch.sigmoid(
        (height_delta + float(below_margin)) / transition
    )
    below_upper_bound = torch.sigmoid(
        (float(above_margin) - height_delta) / transition
    )
    return above_lower_bound * below_upper_bound * supported.to(means.dtype)
