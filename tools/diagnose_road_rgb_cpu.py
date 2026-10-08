"""Probe Road/Background RGB compositing at selected novel-view pixels on CPU.

Approximation: excludes dynamic classes, Sky and appearance affine; reports
remaining transmittance so omitted contributions remain explicit. Uses installed
gsplat PyTorch projection/SH, alpha cutoff, depth ordering and early termination.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def composite(hits):
    hits = sorted(hits, key=lambda h: h[0])
    t = 1.0
    rgb = np.zeros(3)
    weights = {'RoadNodes': 0.0, 'Background': 0.0}
    contributions = {name: np.zeros(3) for name in weights}
    for depth, alpha, color, name in hits:
        color = np.asarray(color)
        next_t = t * (1 - alpha)
        if next_t <= 1e-4:
            break
        w = t * alpha
        weights[name] += w
        contributions[name] += w * color
        rgb += w * color
        t = next_t
    return {'rgb_before_sky_and_affine': rgb.tolist(), 'remaining_transmittance': t,
            'class_weights': weights,
            'class_rgb_contributions': {k: v.tolist() for k, v in contributions.items()},
            'class_weighted_colors': {k: (v/max(weights[k], 1e-12)).tolist() for k, v in contributions.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--gsplat-impl', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=40)
    parser.add_argument('--cam', type=int, default=2)
    parser.add_argument('--left', type=float, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('gsplat_torch_impl', args.gsplat_impl)
    impl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(impl)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    data = args.data
    lidar0 = np.loadtxt(data/'lidar_pose/000000.txt')
    ego0 = np.loadtxt(data/'ego_pose/000000.txt')
    ego = np.linalg.inv(lidar0) @ np.loadtxt(data/f'ego_pose/{args.frame:06d}.txt')
    offset = np.eye(4)
    offset[1, 3] = args.left
    c2w = ego @ offset @ np.linalg.inv(ego0) @ lidar0 @ np.loadtxt(data/f'extrinsics/{args.cam}.txt')
    intr = np.loadtxt(data/f'intrinsics/{args.cam}.txt')
    k = torch.tensor([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]], dtype=torch.float32)[None]
    view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32)[None]
    pixels = [[204.5,500.5], [100.5,450.5], [280.5,470.5], [650.5,450.5]]
    collected = [{name: [] for name in ['learned', 'road_alpha_one']} for _ in pixels]
    isolated = [{name: [] for name in ['learned', 'road_alpha_one']} for _ in pixels]
    for name in ['RoadNodes', 'Background']:
        node = ck['models'][name]
        scales = node['_scales'].exp()
        opacity = node['_opacities'].sigmoid().flatten()
        if name == 'RoadNodes':
            scales[:, :2].clamp_(.02,.08)
            scales[:, 2:].clamp_(.003,.01)
            floor = float(node.get('_opacity_alpha_min', -1))
            opacity = floor+(1-floor)*opacity if floor >= 0 else opacity.clamp(min=.2)
        with torch.no_grad():
            covars, _ = impl._quat_scale_to_covar_preci(node['_quats'], scales, compute_preci=False)
            radii, centers, depths, conics, _ = impl._fully_fused_projection(node['_means'], covars, view, k, 1024, 512, near_plane=.1)
        for pi, pixel in enumerate(pixels):
            delta = centers[0] - torch.tensor(pixel)
            sigma = .5*(conics[0,:,0]*delta[:,0]**2+conics[0,:,2]*delta[:,1]**2)+conics[0,:,1]*delta[:,0]*delta[:,1]
            valid = (radii[0] > 0).all(1) & (sigma >= 0)
            footprint = (-sigma.clamp(min=0)).exp()
            for mode in collected[pi]:
                alpha = footprint * (torch.ones_like(opacity) if name == 'RoadNodes' and mode == 'road_alpha_one' else opacity)
                ids = torch.where(valid & (alpha >= 1/255))[0]
                directions = node['_means'][ids] - torch.tensor(c2w[:3,3], dtype=torch.float32)
                directions = torch.nn.functional.normalize(directions, dim=-1)
                coeffs = torch.cat([node['_features_dc'][ids,None], node['_features_rest'][ids]],dim=1)
                degree = min(int(ck['step'])//1000, int(np.sqrt(coeffs.shape[1]))-1)
                color = (impl._spherical_harmonics(degree, directions, coeffs)+.5).clamp(0,1)
                hits = [(float(depths[0,i]), min(.999,float(alpha[i])), c.tolist(), name) for i,c in zip(ids.tolist(),color)]
                collected[pi][mode].extend(hits)
                if name == 'RoadNodes':
                    isolated[pi][mode].extend(hits)
        print('projected', name, flush=True)
        del covars, radii, centers, depths, conics
    results = []
    for pixel, hits, road_hits in zip(pixels, collected, isolated):
        row = {'pixel':pixel, 'scene':{mode:composite(h) for mode,h in hits.items()},
               'isolated_road':{mode:composite(h) for mode,h in road_hits.items()}}
        results.append(row)
        print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({'checkpoint':str(args.checkpoint), 'limitations':__doc__, 'pixels':results},indent=2))


if __name__ == '__main__':
    main()
