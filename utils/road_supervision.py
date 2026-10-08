"""Losses for road appearance, coverage and contribution, without renderer deps."""
import torch
from models.losses import erode_binary_mask, road_coverage_loss


def supervision_weight(config, step):
    if config is None or float(config.get("w", 0)) <= 0:
        return 0.0
    elapsed = step - int(config.get("start_after", 0))
    if elapsed < 0:
        return 0.0
    warmup = int(config.get("warmup_steps", 0))
    factor = min(1.0, (elapsed + 1) / warmup) if warmup > 0 else 1.0
    return float(config["w"]) * factor


def valid_road_masks(image_infos, erosion_px=3):
    road = image_infos["road_masks"].bool().clone()
    for key in ("egocar_masks", "sky_masks", "dynamic_masks", "human_masks", "vehicle_masks"):
        mask = image_infos.get(key)
        if mask is not None:
            road &= ~mask.bool()
    road = erode_binary_mask(road, erosion_px)
    lane = image_infos.get("lane_masks")
    lane = torch.zeros_like(road) if lane is None else lane.bool() & road
    return road, lane


def unpremultiply_road_rgb(rgb, opacity, eps=1e-4):
    # Keep the denominator differentiable: a single Gaussian's color is
    # independent of its opacity. Coverage receives its own explicit loss.
    return rgb / opacity.clamp_min(eps)


def masked_rgb_l1(prediction, target, mask):
    error = (prediction - target).abs().mean(dim=-1)
    return error[mask].mean() if mask.any() else prediction.sum() * 0


def lane_edge_loss(prediction, target, road, lane):
    # Compare only valid neighboring pairs touching a marking, including its
    # boundary against ordinary asphalt; do not create edges at mask borders.
    total = prediction.sum() * 0
    count = road.new_zeros((), dtype=torch.long)
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = first.copy()
        first[axis], second[axis] = slice(1, None), slice(None, -1)
        first, second = tuple(first), tuple(second)
        valid = road[first] & road[second] & (lane[first] | lane[second])
        delta = (prediction[first] - prediction[second]) - (target[first] - target[second])
        total = total + delta.abs().mean(dim=-1)[valid].sum()
        count = count + valid.sum()
    return total / count.clamp_min(1)


def compute_road_supervision_losses(outputs, image_infos, configs, step):
    losses = {}
    color_cfg = configs.get("road_color")
    coverage_cfg = configs.get("road_coverage")
    contribution_cfg = configs.get("road_contribution")
    mask_cache = {}
    for cfg, kind in ((color_cfg, "color"), (coverage_cfg, "coverage"), (contribution_cfg, "contribution")):
        weight = supervision_weight(cfg, step)
        if weight == 0:
            continue
        required = {"color": "RoadNodes_color", "coverage": "RoadNodes_opacity", "contribution": "RoadNodes_contribution"}[kind]
        if required not in outputs:
            continue
        erosion_px = int(cfg.get("erosion_px", 3))
        if erosion_px not in mask_cache:
            mask_cache[erosion_px] = valid_road_masks(image_infos, erosion_px)
        road, lane = mask_cache[erosion_px]
        if not road.any():
            continue
        if kind == "color":
            prediction = outputs["RoadNodes_color"]
            target = image_infos["pixels"]
            losses["road_color_loss"] = weight * masked_rgb_l1(prediction, target, road)
            if lane.any():
                losses["lane_color_loss"] = weight * float(cfg.get("lane_multiplier", 2)) * masked_rgb_l1(prediction, target, lane)
                losses["lane_edge_loss"] = weight * float(cfg.get("edge_multiplier", 0.2)) * lane_edge_loss(prediction, target, road, lane)
        elif kind == "coverage":
            losses["road_coverage_loss"] = weight * road_coverage_loss(
                outputs["RoadNodes_opacity"], road, float(cfg.get("target_alpha", 0.95)),
                lane_mask=lane, lane_target_alpha=float(cfg.get("lane_target_alpha", 0.98)),
            )
            # A marking's few pixels must not be diluted by the asphalt area.
            if lane.any():
                losses["lane_coverage_loss"] = weight * float(cfg.get("lane_multiplier", 1)) * road_coverage_loss(
                    outputs["RoadNodes_opacity"], lane, float(cfg.get("lane_target_alpha", 0.98)),
                )
        else:
            share = outputs["RoadNodes_contribution"] / outputs["opacity"].clamp_min(1e-4)
            share = share.squeeze(-1)
            losses["road_contribution_loss"] = weight * torch.relu(float(cfg.get("target_share", 0.9)) - share)[road].square().mean()
    return losses


@torch.no_grad()
def road_supervision_metrics(outputs, image_infos, erosion_px=3, target_alpha=0.95,
                             lane_target_alpha=0.98):
    if "RoadNodes_opacity" not in outputs or "road_masks" not in image_infos:
        return {}
    road, lane = valid_road_masks(image_infos, erosion_px)
    metrics = {}
    for name, mask, target in (("road", road, target_alpha), ("lane", lane, lane_target_alpha)):
        metrics[name + "_supervised_pixels"] = mask.sum().float()
        if not mask.any():
            continue
        alpha = outputs["RoadNodes_opacity"].squeeze(-1)[mask]
        metrics[name + "_alpha"] = alpha.mean()
        metrics[name + "_low_coverage_fraction"] = (alpha < target).float().mean()
        if "RoadNodes_color" in outputs:
            metrics[name + "_color_mae"] = masked_rgb_l1(outputs["RoadNodes_color"], image_infos["pixels"], mask)
        if "RoadNodes_contribution" in outputs:
            share = outputs["RoadNodes_contribution"] / outputs["opacity"].clamp_min(1e-4)
            metrics[name + "_contribution_share"] = share.squeeze(-1)[mask].mean()
    return metrics
