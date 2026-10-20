"""Multi-view image-edge guided, fixed-surface RoadNodes initialization.

No CUDA renderer, lane labels, global ground plane, or training-time growth is
required. Candidates are interpolated on the existing road lattice and tested
by projection into *training* views before they are allowed into the model.
"""
import json
import logging
import math
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _frame_mask(camera, name, frame, shape, device):
    value = getattr(camera, name, None)
    if value is None:
        return torch.zeros(shape, dtype=torch.bool, device=device)
    value = value[frame] if value.ndim == 3 else value
    if tuple(value.shape) != tuple(shape):
        raise ValueError("{} must match loaded image resolution".format(name))
    return value.to(device=device, dtype=torch.bool)


@torch.no_grad()
def image_edge_evidence(camera, frame, config, device):
    """Return a dilated RGB edge band strictly inside unoccluded road masks."""
    rgb = camera.images[frame].to(device=device, dtype=torch.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("High-frequency initialization requires HWC RGB images")
    h, w = rgb.shape[:2]
    if getattr(camera, "road_masks", None) is None:
        raise ValueError("High-frequency initialization requires road_masks")
    valid = _frame_mask(camera, "road_masks", frame, (h, w), device)
    for name in ("egocar_mask", "sky_masks", "dynamic_masks", "human_masks", "vehicle_masks"):
        valid = valid & ~_frame_mask(camera, name, frame, (h, w), device)

    # Keep invalid pixels outside the gradient and edge-band support. Using a
    # wider erosion than the blur radius also excludes segmentation boundaries.
    erosion = int(config.get("erosion_px", 4))
    if erosion < 4:
        raise ValueError("hf erosion_px must be >= 4 for the 7px coarse filter")
    kernel = 2 * erosion + 1
    valid = F.avg_pool2d(valid.float()[None, None], kernel, 1, erosion)[0, 0] > 1 - 1e-6
    image = rgb.permute(2, 0, 1)[None]

    def gradient(window):
        blurred = F.avg_pool2d(image, window, 1, window // 2)
        gx = F.pad((blurred[..., 2:] - blurred[..., :-2]).abs().mean(1), (1, 1, 0, 0))
        gy = F.pad((blurred[..., 2:, :] - blurred[..., :-2, :]).abs().mean(1), (0, 0, 1, 1))
        return torch.maximum(gx, gy)[0]

    fine, coarse = gradient(3), gradient(7)
    fine_min = float(config.get("fine_gradient_min", .04))
    coarse_min = float(config.get("coarse_gradient_min", .015))
    if fine_min <= 0 or coarse_min <= 0:
        raise ValueError("hf gradient thresholds must be positive")
    strength = torch.where(valid & (fine >= fine_min) & (coarse >= coarse_min),
                           fine, torch.zeros_like(fine))
    radius = int(config.get("edge_band_px", 3))
    if radius < 0:
        raise ValueError("edge_band_px must be nonnegative")
    if radius:
        strength = F.max_pool2d(strength[None, None], 2 * radius + 1, 1, radius)[0, 0]
    return rgb, valid, strength * valid


def training_views(cameras, frame_indices, config):
    """Interleave cameras over time; reject repeated near-identical poses."""
    if not cameras:
        raise ValueError("High-frequency initialization requires training cameras")
    entries = sorted(cameras.items(), key=lambda item: str(item[0]))
    lengths = [len(camera.images) for _, camera in entries]
    if frame_indices is None:
        frames = list(range(min(lengths)))
    else:
        frames = sorted(set(int(i) for i in frame_indices))
        if any(i < 0 or i >= min(lengths) for i in frames):
            raise ValueError("hf frame indices must be local training-frame indices")
    stride = int(config.get("frame_stride", 10))
    limit = int(config.get("max_views", 60))
    baseline = float(config.get("min_view_baseline", .5))
    if stride < 1 or limit < 1 or baseline < 0:
        raise ValueError("Invalid hf view sampling configuration")
    sampled = frames[::stride]
    if frames and frames[-1] not in sampled:
        sampled.append(frames[-1])
    pairs = [(key, camera, frame) for frame in sampled for key, camera in entries]
    # Uniformly cap the timeline rather than silently keeping only early frames.
    if len(pairs) > limit:
        ids = torch.linspace(0, len(pairs) - 1, limit).round().long().tolist()
        pairs = [pairs[i] for i in ids]
    accepted, last_origins = [], {}
    for key, camera, frame in pairs:
        origin = camera.cam_to_worlds[frame, :3, 3].detach().cpu()
        if key in last_origins and float((origin - last_origins[key]).norm()) < baseline:
            continue
        last_origins[key] = origin
        accepted.append((key, camera, frame))
    return accepted


@torch.no_grad()
def project_points(points, camera, frame, config):
    """Chunked pinhole projection in the same coordinates as loaded cameras."""
    h, w = camera.images.shape[1:3]
    pose = camera.cam_to_worlds[frame].to(points)
    intrinsics = camera.intrinsics[frame].to(points)
    world_to_cam = torch.linalg.inv(pose)
    chunk = int(config.get("projection_chunk_size", 131072))
    if chunk < 1:
        raise ValueError("projection_chunk_size must be positive")
    pixels = torch.zeros(len(points), dtype=torch.long, device=points.device)
    depths = points.new_zeros(len(points))
    inside = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for start in range(0, len(points), chunk):
        end = min(start + chunk, len(points))
        xyz = points[start:end] @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
        uv = xyz @ intrinsics.T
        uv = uv[:, :2] / uv[:, 2:].clamp_min(1e-6)
        finite = torch.isfinite(xyz).all(-1) & torch.isfinite(uv).all(-1)
        x, y = uv[:, 0].round().long(), uv[:, 1].round().long()
        inside[start:end] = (finite & (xyz[:, 2] > .1)
                             & (xyz[:, 2] <= float(config.get("max_depth", 60)))
                             & (x >= 0) & (x < w) & (y >= 0) & (y < h))
        pixels[start:end] = y.clamp(0, h - 1) * w + x.clamp(0, w - 1)
        depths[start:end] = xyz[:, 2]
    return pixels, depths, inside


@torch.no_grad()
def visible_road_points(points, camera, frame, config, valid_road, reference=None):
    """Mask occluders, road self-occlusion, and available sparse LiDAR depth."""
    h, w = valid_road.shape
    pixel, depth, inside = project_points(points, camera, frame, config)
    zbuffer = points.new_full((h * w,), float('inf'))
    reference_projection = (pixel, depth, inside) if reference is None else project_points(reference, camera, frame, config)
    rp, rz, ri = reference_projection
    zbuffer.scatter_reduce_(0, rp[ri], rz[ri], reduce="amin", include_self=True)
    # Neighboring pixels help a coarse road lattice support finer candidates.
    zbuffer = -F.max_pool2d(-zbuffer.reshape(1, 1, h, w), 3, 1, 1).flatten()
    tolerance = float(config.get("depth_tolerance", .25)) + depth.clamp_min(0) * float(config.get("relative_depth_tolerance", .02))
    visible = inside & valid_road.flatten()[pixel] & (depth <= zbuffer[pixel] + tolerance)
    sparse = getattr(camera, "lidar_depth_maps", None)
    if sparse is not None:
        sparse = sparse[frame].to(points).reshape(h * w)[pixel]
        valid_sparse = torch.isfinite(sparse) & (sparse > 0)
        visible &= ~valid_sparse | (depth <= sparse + tolerance)
    return pixel, visible


@torch.no_grad()
def accumulate_evidence(points, views, config, reference=None, collect_colors=False):
    observations = points.new_zeros(len(points))
    support = points.new_zeros(len(points))
    strengths = points.new_zeros(len(points))
    previews = []
    max_samples = int(config.get("color_samples", 7))
    if max_samples < 1:
        raise ValueError("color_samples must be positive")
    samples = points.new_full((len(points), max_samples, 3), float('nan')) if collect_colors else None
    sample_counts = torch.zeros(len(points), dtype=torch.long, device=points.device) if collect_colors else None
    for index, (key, camera, frame) in enumerate(views):
        rgb, valid, edge = image_edge_evidence(camera, frame, config, points.device)
        pixel, visible = visible_road_points(points, camera, frame, config, valid, reference)
        evidence = edge.flatten()[pixel]
        supported = visible & (evidence > 0)
        observations += visible
        support += supported
        strengths += evidence * supported
        if collect_colors:
            ids = torch.where(visible & (sample_counts < max_samples))[0]
            samples[ids, sample_counts[ids]] = rgb.reshape(-1, 3)[pixel[ids]]
            sample_counts[ids] += 1
        elif len(previews) < int(config.get("preview_views", 5)):
            ids = torch.where(visible)[0]
            previews.append((str(key), frame, rgb.cpu(), edge.cpu(), ids.cpu(), pixel[ids].cpu()))
        if index % 10 == 0 or index + 1 == len(views):
            logger.info("Road HF %s: view %d/%d camera=%s frame=%d",
                        "color fusion" if collect_colors else "evidence", index + 1, len(views), key, frame)
    ratio = support / observations.clamp_min(1)
    scores = ratio * strengths / support.clamp_min(1)
    accepted = (support >= int(config.get("min_support_views", 3))) & (ratio >= float(config.get("min_support_ratio", .35)))
    colors = torch.nanmedian(samples, dim=1).values if collect_colors else None
    return dict(observations=observations, support=support, scores=scores,
                accepted=accepted, colors=colors, previews=previews)


@torch.no_grad()
def lattice_candidates(means, colors, anchors, spacing=.05, height_tolerance=.05):
    """Insert three half-cell samples only in fully supported, continuous cells."""
    if spacing <= 0 or height_tolerance <= 0:
        raise ValueError("Invalid hf lattice geometry")
    origin = means[:, :2].amin(0)
    cells = ((means[:, :2] - origin) / spacing).round().long()
    if not torch.allclose(origin + cells.to(means) * spacing, means[:, :2], atol=1e-4, rtol=0):
        raise ValueError("HF initialization requires the existing regular road lattice")
    width = int(cells[:, 0].max()) + 1
    height = int(cells[:, 1].max()) + 1
    keys = cells[:, 1] * width + cells[:, 0]
    sorted_keys, order = keys.sort()
    if len(torch.unique_consecutive(sorted_keys)) != len(keys):
        raise ValueError("Duplicate base lattice centers")
    ac = cells[anchors]
    neighbors = []
    valid = (ac[:, 0] + 1 < width) & (ac[:, 1] + 1 < height)
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        query = (ac[:, 1] + dy) * width + ac[:, 0] + dx
        loc = torch.searchsorted(sorted_keys, query).clamp_max(len(keys) - 1)
        valid &= sorted_keys[loc] == query
        neighbors.append(order[loc])
    neighbors = torch.stack(neighbors, -1)
    z = means[neighbors, 2]
    valid &= z.amax(-1) - z.amin(-1) <= height_tolerance
    neighbors, anchors = neighbors[valid], anchors[valid]
    weights = means.new_tensor([[.5, .5, 0, 0], [.5, 0, .5, 0], [.25, .25, .25, .25]])
    points = torch.einsum('kc,ncd->nkd', weights, means[neighbors]).reshape(-1, 3)
    rgb = torch.einsum('kc,ncd->nkd', weights, colors[neighbors]).reshape(-1, 3)
    return points, rgb, anchors.repeat_interleave(3)


@torch.no_grad()
def initialize_high_frequency_road(means, colors, cameras, frames, config, output_dir=None):
    """Return additional fixed centers/colors plus reviewable initialization data."""
    ratio = float(config.get("max_extra_ratio", .25))
    if not 0 <= ratio <= 1:
        raise ValueError("max_extra_ratio must be in [0, 1]")
    min_views = int(config.get("min_support_views", 3))
    support_ratio = float(config.get("min_support_ratio", .35))
    if min_views < 2 or not 0 < support_ratio <= 1:
        raise ValueError("HF initialization requires >=2 views and support ratio in (0, 1]")
    views = training_views(cameras, frames, config)
    budget = math.floor(len(means) * ratio)
    evidence = accumulate_evidence(means, views, config)
    eligible = torch.where(evidence['accepted'])[0]
    # Oversample cells slightly: incomplete cells or unsupported child locations
    # may subsequently be rejected, but the final point budget is always hard.
    anchor_count = min(len(eligible), math.ceil(2 * budget / 3))
    anchors = eligible[evidence['scores'][eligible].topk(anchor_count).indices] if anchor_count else eligible[:0]
    added, added_colors, parents = lattice_candidates(
        means, colors, anchors,
        spacing=float(config.get("base_spacing", .05)),
        height_tolerance=float(config.get("height_tolerance", .05)))
    candidate_count = len(added)
    if candidate_count:
        child = accumulate_evidence(added, views, config, reference=means, collect_colors=True)
        valid = torch.where(child['accepted'] & torch.isfinite(child['colors']).all(-1))[0]
        if len(valid) > budget:
            valid = valid[child['scores'][valid].topk(budget).indices]
        added, added_colors, parents = added[valid], child['colors'][valid].clamp(0, 1), parents[valid]
    stats = dict(base_points=len(means), views=len(views), eligible_anchors=len(eligible),
                 proposed_points=candidate_count, added_points=len(added), budget=budget,
                 extra_ratio=len(added) / max(len(means), 1), base_spacing=float(config.get('base_spacing', .05)),
                 detail_spacing=float(config.get('base_spacing', .05)) / 2,
                 min_support_views=min_views, min_support_ratio=support_ratio)
    logger.info("Road HF initialization: %s", stats)
    if not len(added):
        logger.warning("Road HF initialization selected no supported details; retaining base road unchanged")
    if output_dir is not None:
        write_diagnostics(output_dir, means, added, added_colors, parents, anchors, evidence, stats)
    return added, added_colors, stats


def write_diagnostics(output_dir, means, added, colors, parents, anchors, evidence, stats):
    import numpy as np
    from PIL import Image, ImageDraw
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'summary.json').write_text(json.dumps(stats, indent=2))
    base = means.detach().cpu().numpy()
    extra = added.detach().cpu().numpy()
    selected = anchors.detach().cpu().numpy()
    scores = evidence['scores'].detach().cpu().numpy()
    np.savez_compressed(root / 'details.npz', means=extra, colors=colors.cpu().numpy(),
                        parent_indices=parents.cpu().numpy(), selected_anchors=base[selected],
                        selected_scores=scores[selected])
    lo, hi = base[:, :2].min(0), base[:, :2].max(0)
    scale = 1535 / max(float((hi - lo).max()), .01)
    shape = np.maximum(1, np.ceil((hi - lo) * scale).astype(int) + 1)
    xy = np.clip(((base[:, :2] - lo) * scale).astype(int), 0, shape - 1)
    intensity = np.zeros((shape[1], shape[0]), dtype=np.float32)
    np.maximum.at(intensity, (xy[:, 1], xy[:, 0]), scores)
    intensity /= max(float(intensity.max()), 1e-6)
    Image.fromarray(np.uint8(intensity * 255)).save(root / 'edge_evidence_bev.png')
    density = np.zeros((shape[1], shape[0], 3), dtype=np.uint8)
    density[xy[:, 1], xy[:, 0]] = (80, 80, 80)
    if len(extra):
        exy = np.clip(((extra[:, :2] - lo) * scale).astype(int), 0, shape - 1)
        density[exy[:, 1], exy[:, 0]] = (255, 110, 40)
    Image.fromarray(density).save(root / 'detail_density_bev.png')
    chosen = np.zeros(len(base), dtype=bool)
    if len(parents):
        chosen[parents.cpu().numpy()] = True
    for key, frame, rgb, edge, ids, pixels in evidence['previews']:
        image = np.uint8(rgb.numpy().clip(0, 1) * 255)
        band = edge.numpy() > 0
        image[band] = np.uint8(.5 * image[band] + .5 * np.array([255, 60, 60]))
        overlay = Image.fromarray(image)
        draw = ImageDraw.Draw(overlay)
        flat = pixels.numpy()[chosen[ids.numpy()]]
        h, w = image.shape[:2]
        for pix in flat[::max(1, len(flat) // 4000)]:
            x, y = int(pix % w), int(pix // w)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(40, 255, 80))
        draw.text((8, 8), 'red: RGB edge band; green: accepted detail parents', fill='white')
        overlay.save(root / 'camera_{}_frame_{:06d}.jpg'.format(key, frame))
