import torch

from utils.road_surface_prior import (
    build_road_surface_grid,
    road_intrusion_confidence,
)


def test_road_intrusion_confidence_is_local_and_height_aware():
    road_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )
    origin, heights, valid, cell_size = build_road_surface_grid(
        road_points, cell_size=0.5
    )
    means = torch.tensor(
        [
            [0.1, 0.1, 0.0],
            [0.1, 0.1, 1.0],
            [2.0, 2.0, 0.0],
        ]
    )
    confidence = road_intrusion_confidence(
        means,
        origin,
        heights,
        valid,
        cell_size,
        below_margin=0.1,
        above_margin=0.15,
        transition=0.05,
    )

    assert confidence[0] > 0.8
    assert confidence[1] < 1e-6
    assert confidence[2] == 0


def test_road_surface_grid_respects_memory_bound():
    road_points = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 1.0]])
    _, heights, _, effective_cell_size = build_road_surface_grid(
        road_points, cell_size=0.1, max_grid_cells=10_000
    )

    assert heights.numel() <= 10_000
    assert effective_cell_size > 0.1


def test_detached_confidence_only_updates_opacity():
    road_points = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    origin, heights, valid, cell_size = build_road_surface_grid(
        road_points, cell_size=0.5
    )
    means = torch.tensor([[0.1, 0.1, 0.0]], requires_grad=True)
    opacity_logits = torch.zeros((1,), requires_grad=True)
    confidence = road_intrusion_confidence(
        means,
        origin,
        heights,
        valid,
        cell_size,
        below_margin=0.1,
        above_margin=0.15,
        transition=0.05,
    ).detach()
    loss = (torch.sigmoid(opacity_logits) * confidence).sum()
    loss.backward()

    assert means.grad is None
    assert opacity_logits.grad is not None
    assert opacity_logits.grad.item() > 0


if __name__ == "__main__":
    test_road_intrusion_confidence_is_local_and_height_aware()
    test_road_surface_grid_respects_memory_bound()
    test_detached_confidence_only_updates_opacity()
