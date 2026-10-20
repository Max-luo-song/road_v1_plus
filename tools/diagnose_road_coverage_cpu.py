"""CPU footprint diagnostic; does not replace an exact CUDA render.

Uses gsplat's PyTorch projection and alpha cutoff, with sampled pixels and
order-independent transmittance accumulation. Ignores early termination, RGB,
other classes, lens distortion and ego compositing. Novel renders use pinhole
projection before image distortion, matching this diagnostic's coordinate space.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from utils.road_scales import decode_road_scales


def accumulate(means2d, conics, radii, opacity, width, height, stride):
    xs = np.arange(0, width, stride) + 0.5
    ys = np.arange(0, height, stride) + 0.5
    log_t = np.zeros((len(ys), len(xs)), dtype=np.float64)
    counts = np.zeros_like(log_t, dtype=np.int32)
    for center, conic, radius, alpha in zip(means2d, conics, radii, opacity):
        if min(radius) <= 0:
            continue
        x0, x1 = np.searchsorted(xs, [center[0] - radius[0], center[0] + radius[0]])
        y0, y1 = np.searchsorted(ys, [center[1] - radius[1], center[1] + radius[1]])
        if x1 <= x0 or y1 <= y0:
            continue
        dx = xs[x0:x1][None, :] - center[0]
        dy = ys[y0:y1][:, None] - center[1]
        sigma = 0.5 * (conic[0] * dx**2 + conic[2] * dy**2) + conic[1] * dx * dy
        contribution = np.minimum(0.999, alpha * np.exp(-np.maximum(sigma, 0)))
        active = (sigma >= 0) & (contribution >= 1 / 255)
        contribution = np.where(active, contribution, 0)
        log_t[y0:y1, x0:x1] += np.log1p(-contribution)
        counts[y0:y1, x0:x1] += active
    return -np.expm1(log_t), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gsplat-impl', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=40)
    parser.add_argument('--cam', type=int, default=2)
    parser.add_argument('--left', type=float, default=3)
    parser.add_argument('--stride', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('gsplat_torch_projection', args.gsplat_impl)
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)
    road = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)['models']['RoadNodes']
    data = args.data
    lidar0 = np.loadtxt(data / 'lidar_pose/000000.txt')
    ego0 = np.loadtxt(data / 'ego_pose/000000.txt')
    ego = np.linalg.inv(lidar0) @ np.loadtxt(data / f'ego_pose/{args.frame:06d}.txt')
    offset = np.eye(4)
    offset[1, 3] = args.left
    cam_to_ego = np.linalg.inv(ego0) @ lidar0 @ np.loadtxt(data / f'extrinsics/{args.cam}.txt')
    c2w = ego @ offset @ cam_to_ego
    intrinsics = np.loadtxt(data / f'intrinsics/{args.cam}.txt')
    image_path = next((data / 'images').glob(f'{args.frame:06d}_{args.cam}.*'))
    width, height = Image.open(image_path).size
    k = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
    view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32)[None]
    k = torch.tensor(k, dtype=torch.float32)[None]
    alpha_min = float(road.get('_opacity_alpha_min', -1))
    alpha = road['_opacities'].sigmoid().flatten()
    alpha = alpha_min + (1 - alpha_min) * alpha if alpha_min >= 0 else alpha.clamp(min=0.2)
    scales = decode_road_scales(road['_scales'], road.get('_scale_xy_bounds', (-1., -1.)))
    saved_bounds = tuple(float(v) for v in road.get('_scale_xy_bounds', (-1., -1.)))
    xy_min, xy_max = saved_bounds if saved_bounds[0] > 0 else (.02, .08)
    results = {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, forced_alpha, scale_multiplier in [('learned', False, 1), ('alpha_one', True, 1), ('alpha_one_xy_double', True, 2)]:
        test_scales = scales.clone()
        test_scales[:, :2] *= scale_multiplier
        with torch.no_grad():
            covars, _ = impl._quat_scale_to_covar_preci(road['_quats'], test_scales, compute_preci=False)
            radii, centers, _, conics, _ = impl._fully_fused_projection(road['_means'], covars, view, k, width, height, near_plane=0.1)
        a = np.ones(len(alpha)) if forced_alpha else alpha.numpy()
        # The 3.33-sigma box is deliberately generous for low-opacity points;
        # the per-pixel cutoff removes their extra zero contributions.
        opacity, counts = accumulate(centers[0].numpy(), conics[0].numpy(), radii[0].numpy(), a, width, height, args.stride)
        np.savez_compressed(args.output / f'{name}.npz', alpha=opacity, counts=counts)
        Image.fromarray(np.uint8(opacity * 255)).resize((width, height)).save(args.output / f'{name}.png')
        roi = opacity[len(opacity)//2:, :len(opacity[0])//3]
        results[name] = {'bottom_left_alpha_mean': float(roi.mean()), 'bottom_left_alpha_below_0.5': float((roi < 0.5).mean()), 'bottom_left_zero_contributors': float((counts[len(counts)//2:, :len(counts[0])//3] == 0).mean())}
        # Separate the near road rectangle from the horizon/non-road part above.
        near_roi = opacity[int(height * 0.78125)//args.stride:, :width//3//args.stride]
        results[name]['near_road'] = {'mean_alpha': float(near_roi.mean()), 'below_0.5': float((near_roi < 0.5).mean()), 'below_0.95': float((near_roi < 0.95).mean())}
        if name == 'learned':
            local_stats = []
            raw = road['_opacities'].flatten().numpy()
            physical_scale = scales.numpy()
            center_np, conic_np, radius_np = centers[0].numpy(), conics[0].numpy(), radii[0].numpy()
            for flat_idx in np.argsort(near_roi, axis=None)[:12]:
                iy, ix = np.unravel_index(flat_idx, near_roi.shape)
                x, y = ix * args.stride + 0.5, (iy + int(height * 0.78125)//args.stride) * args.stride + 0.5
                delta = center_np - [x, y]
                sigma = 0.5 * (conic_np[:, 0] * delta[:, 0]**2 + conic_np[:, 2] * delta[:, 1]**2) + conic_np[:, 1] * delta[:, 0] * delta[:, 1]
                footprint = np.exp(-np.maximum(sigma, 0))
                valid = (radius_np > 0).all(1) & (sigma >= 0)
                current = np.where(valid, a * footprint, 0)
                potential = np.where(valid, footprint, 0)
                neighbors = potential >= 1/255
                local_stats.append({'pixel': [float(x), float(y)], 'alpha': float(near_roi[iy, ix]), 'potential_contributors': int(neighbors.sum()), 'fraction_neighbor_raw_below_minus14': float((raw[neighbors] < -14).mean()), 'fraction_neighbor_xy_both_near_bounds': float(((physical_scale[neighbors, :2] <= xy_min * 1.001) | (physical_scale[neighbors, :2] >= xy_max * .999)).all(1).mean()), 'neighbor_alpha_mean': float(a[neighbors].mean()), 'alpha_if_all_one': float(1-np.prod(1-np.minimum(.999, potential[neighbors])))})
            results['leaking_pixel_neighbors'] = local_stats
        print(name, results[name], flush=True)
    (args.output / 'summary.json').write_text(json.dumps({'checkpoint': str(args.checkpoint), 'frame': args.frame, 'cam': args.cam, 'left': args.left, 'stride': args.stride, 'space': 'pinhole before image distortion; bottom-left rectangle is not a road mask', 'results': results}, indent=2))


if __name__ == '__main__':
    main()
