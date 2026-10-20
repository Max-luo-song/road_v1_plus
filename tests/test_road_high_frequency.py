import ast
import logging
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from utils.road_high_frequency import (
    accumulate_evidence, image_edge_evidence, initialize_high_frequency_road,
    lattice_candidates, project_points, training_views, visible_road_points,
)
import test_road_adaptive_footprint as footprint_tests
from test_road_adaptive_footprint import road_class


def config(**overrides):
    cfg = dict(enabled=True, frame_stride=1, max_views=20, min_view_baseline=.05,
               min_support_views=3, min_support_ratio=.5, fine_gradient_min=.04,
               coarse_gradient_min=.015, erosion_px=4, edge_band_px=3,
               base_spacing=.05, max_extra_ratio=.25, detail_scale=.015,
               detail_opacity=.1, detail_warmup_steps=1000, preview_views=2)
    cfg.update(overrides)
    return cfg


def road_grid(spacing=.05):
    a = torch.arange(-.4, .401, spacing)
    xy = torch.cartesian_prod(a, a)
    return torch.cat((xy, torch.zeros(len(xy), 1)), -1)


def camera(stripe=True):
    size, focal = 128, 80.
    poses = torch.eye(4).repeat(3, 1, 1)
    poses[:, :3, :3] = torch.diag(torch.tensor([1., -1., -1.]))
    poses[:, 2, 3] = 1.
    poses[:, 0, 3] = torch.tensor([0., .1, .2])
    intrinsics = torch.tensor([[focal, 0, 64.], [0, focal, 64.], [0, 0, 1.]]).repeat(3, 1, 1)
    images = torch.full((3, size, size, 3), .2)
    if stripe:
        for i in range(3):
            world_x = (torch.arange(size) - 64) / focal + poses[i, 0, 3]
            images[i, :, world_x.abs() < .04] = .9
    return SimpleNamespace(images=images, intrinsics=intrinsics, cam_to_worlds=poses,
                           road_masks=torch.ones(3, size, size, dtype=torch.bool),
                           egocar_mask=torch.zeros(size, size, dtype=torch.bool),
                           dynamic_masks=torch.zeros(3, size, size, dtype=torch.bool),
                           lidar_depth_maps=None)


class RoadHighFrequencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flat_images_do_not_grow_points_or_mask_boundary_edges(self):
        cam = camera(False)
        cam.road_masks[:, :20] = False
        cam.images[:, :20] = 1.
        _, _, edge = image_edge_evidence(cam, 0, config(), torch.device('cpu'))
        self.assertFalse(edge.any())
        points = road_grid()
        added, _, stats = initialize_high_frequency_road(points, torch.full_like(points, .4), {2: cam}, [0, 1, 2], config())
        self.assertEqual(len(added), 0)
        self.assertEqual(stats['views'], 3)

    def test_repeated_world_edges_grow_local_points_with_hard_budget(self):
        points = road_grid()
        with tempfile.TemporaryDirectory() as tmp:
            added, colors, stats = initialize_high_frequency_road(points, torch.full_like(points, .4), {2: camera()}, [0, 1, 2], config(), tmp)
            self.assertGreater(len(added), 0)
            self.assertLessEqual(len(added), int(len(points) * .25))
            self.assertTrue((added[:, 0].abs() < .15).all())
            self.assertTrue((added[:, 2] == 0).all())
            self.assertTrue(torch.isfinite(colors).all())
            self.assertEqual(len(torch.unique(torch.cat((points, added)).round(decimals=4), dim=0)), len(points) + len(added))
            self.assertEqual(stats['detail_spacing'], .025)
            for filename in ('summary.json', 'details.npz', 'edge_evidence_bev.png', 'detail_density_bev.png', 'camera_2_frame_000000.jpg'):
                self.assertTrue((Path(tmp) / filename).is_file())

    def test_single_view_and_static_duplicate_poses_are_not_consensus(self):
        points = road_grid()
        for frames, static in [([0], False), ([0, 1, 2], True)]:
            cam = camera()
            if static:
                cam.cam_to_worlds[:] = cam.cam_to_worlds[0].clone()
            added, _, stats = initialize_high_frequency_road(points, torch.full_like(points, .4), {2: cam}, frames, config())
            self.assertEqual(len(added), 0)
            self.assertEqual(stats['views'], 1)

    def test_held_out_frames_are_excluded(self):
        cam = camera()
        views = training_views({2: cam}, [0, 2], config(min_view_baseline=0))
        self.assertEqual([v[2] for v in views], [0, 2])

    def test_dynamic_occlusion_excludes_edges(self):
        cam = camera()
        for i in range(3):
            world_x = (torch.arange(128) - 64) / 80 + cam.cam_to_worlds[i, 0, 3]
            cam.dynamic_masks[i, :, world_x.abs() < .2] = True
        points = road_grid()
        added, _, _ = initialize_high_frequency_road(points, torch.full_like(points, .4), {2: cam}, [0, 1, 2], config())
        self.assertEqual(len(added), 0)

    def test_road_self_occlusion_and_sparse_depth(self):
        cam = camera(False)
        points = torch.tensor([[0., 0, 0], [0., 0, -.5]])
        valid = torch.ones(128, 128, dtype=torch.bool)
        _, visible = visible_road_points(points, cam, 0, config(), valid)
        self.assertEqual(visible.tolist(), [True, False])
        cam.lidar_depth_maps = torch.full((3, 128, 128), .5)
        _, visible = visible_road_points(points, cam, 0, config(), valid)
        self.assertFalse(visible.any())
        cam.lidar_depth_maps[:] = float('nan')
        _, visible = visible_road_points(points, cam, 0, config(), valid)
        self.assertEqual(visible.tolist(), [True, False])

    def test_candidates_follow_local_slope_and_reject_holes_or_steps(self):
        means = torch.tensor([[0., 0, 0], [.05, 0, .005], [0, .05, .01], [.05, .05, .015]])
        colors = torch.full_like(means, .4)
        points, _, _ = lattice_candidates(means, colors, torch.tensor([0]))
        self.assertEqual(len(points), 3)
        torch.testing.assert_close(points[:, 2], .1 * points[:, 0] + .2 * points[:, 1])
        hole, _, _ = lattice_candidates(means[:3], colors[:3], torch.tensor([0]))
        self.assertEqual(len(hole), 0)
        means[-1, 2] = .2
        step, _, _ = lattice_candidates(means, colors, torch.tensor([0]))
        self.assertEqual(len(step), 0)

    def test_color_median_rejects_single_view_outlier(self):
        cam = camera(False)
        cam.images[0] = .4
        cam.images[1] = .4
        cam.images[2] = .95
        views = training_views({2: cam}, [0, 1, 2], config())
        result = accumulate_evidence(torch.tensor([[0., 0, 0]]), views, config(), collect_colors=True)
        torch.testing.assert_close(result['colors'], torch.full((1, 3), .4))

    def test_projection_chunk_size_does_not_change_selected_pixels(self):
        points, cam = road_grid(), camera()
        small = project_points(points, cam, 1, config(projection_chunk_size=3))
        large = project_points(points, cam, 1, config(projection_chunk_size=65536))
        for a, b in zip(small, large):
            torch.testing.assert_close(a, b)

    def test_trainer_passes_training_cameras_frames_and_output_directory(self):
        tree = ast.parse(Path('models/trainers/scene_graph.py').read_text())
        source = Path('models/trainers/scene_graph.py').read_text()
        block = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Compare)
                     and isinstance(node.test.left, ast.Name) and node.test.left.id == 'class_name'
                     and isinstance(node.test.comparators[0], ast.Constant)
                     and node.test.comparators[0].value == 'RoadNodes'
                     and 'create_from_pcd' in ast.get_source_segment(source, node))
        received = {}
        def create(**kwargs):
            received.update(kwargs)
            return False
        cameras, frames = {2: camera()}, [0, 2]
        dataset = SimpleNamespace(pixel_source=SimpleNamespace(camera_data=cameras), train_timesteps=frames)
        env = dict(class_name='RoadNodes', model=SimpleNamespace(ctrl_cfg={}, create_from_pcd=create),
                   processed_init_road_pts={'pts': road_grid(), 'colors': torch.full_like(road_grid(), .4)},
                   dataset=dataset, initialization_output_dir='/tmp/hf-test-output', logger=logging.getLogger())
        exec(compile(ast.Module(body=[block], type_ignores=[]), 'road_init_entry', 'exec'), env)
        self.assertIs(received['road_cameras'], cameras)
        self.assertIs(received['road_train_frames'], frames)
        self.assertEqual(received['initialization_output_dir'], '/tmp/hf-test-output')

    def test_train_entry_uses_configured_group_directory_without_setup_locals(self):
        tree = ast.parse(Path('tools/train.py').read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        branch = next(n for n in main.body if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Compare)
                      and isinstance(n.test.left, ast.Attribute) and n.test.left.attr == 'resume_from')
        received = {}
        def initialize(**kwargs):
            received.update(kwargs)
        dataset = object()
        env = dict(cfg=SimpleNamespace(log_dir='/tmp/run/group0'), os=os, dataset=dataset,
                   trainer=SimpleNamespace(init_gaussians_from_dataset=initialize, step=0),
                   logger=logging.getLogger())
        # Execute the actual fresh-training branch without importing CUDA-only
        # modules or supplying setup()'s local log_dir variable.
        exec(compile(ast.Module(body=branch.orelse, type_ignores=[]), 'train_fresh_entry', 'exec'), env)
        self.assertIs(received['dataset'], dataset)
        self.assertEqual(received['initialization_output_dir'], '/tmp/run/group0/road_init')

    def test_old_checkpoint_load_does_not_silently_reinitialize_details(self):
        Road = road_class()
        model = Road('RoadNodes', ctrl=OmegaConf.create({'road_sh_degree': 0, 'smooth_xy_scale': True}),
                     reg=OmegaConf.create({'alpha_min': .01, 'lowest_opacity': .01}),
                     init={'high_frequency': config()}, device=torch.device('cpu'))
        with self.assertLogs(level='WARNING') as messages:
            model.load_state_dict(footprint_tests.AdaptiveFootprintTests.checkpoint(True), strict=True)
        self.assertEqual(model.num_points, 2)
        self.assertFalse(model._hf_detail_mask.any())
        self.assertIn('only runs when training from scratch', '\n'.join(messages.output))

    def test_real_road_initialization_preserves_base_and_restores_details(self):
        Road = road_class()
        def make(hf):
            return Road('RoadNodes', ctrl=OmegaConf.create({'road_sh_degree': 0, 'smooth_xy_scale': True}),
                        reg=OmegaConf.create({'alpha_min': .01, 'lowest_opacity': .01}),
                        init={'opacity_init_value': .7, 'high_frequency': config(enabled=hf)},
                        device=torch.device('cpu'))
        points = road_grid(.1)
        colors = torch.full_like(points, .4)
        base, model = make(False), make(True)
        base.create_from_pcd(points, colors)
        model.create_from_pcd(points, colors, road_cameras={2: camera()}, road_train_frames=[0, 1, 2])
        self.assertGreater(int(model._hf_detail_mask.sum()), 0)
        torch.testing.assert_close(model._means[:base.num_points], base._means, rtol=0, atol=0)
        torch.testing.assert_close(model.get_scaling[:base.num_points], base.get_scaling)
        torch.testing.assert_close(model.get_opacity[:base.num_points], base.get_opacity)
        torch.testing.assert_close(model.get_scaling[model._hf_detail_mask, :2], torch.full((int(model._hf_detail_mask.sum()), 2), .015))
        torch.testing.assert_close(model.get_opacity[model._hf_detail_mask], torch.full((int(model._hf_detail_mask.sum()), 1), .1))
        # Adam must leave still-locked details untouched, not cancel a warmup
        # multiplier through first/second-moment normalization.
        raw = model._opacities.detach().clone()
        optim = torch.optim.Adam([model._opacities], lr=.01)
        model._opacities.sum().backward()
        optim.step()
        torch.testing.assert_close(model._opacities[model._hf_detail_mask], raw[model._hf_detail_mask], rtol=0, atol=0)
        optim.zero_grad()
        model.step = 1000
        model._opacities.sum().backward()
        optim.step()
        self.assertFalse(torch.equal(model._opacities[model._hf_detail_mask], raw[model._hf_detail_mask]))
        restored = make(True)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored._hf_detail_mask, model._hf_detail_mask)
        torch.testing.assert_close(restored.get_scaling, model.get_scaling)
        torch.testing.assert_close(restored.get_opacity, model.get_opacity)
        means = restored._means.detach().clone()
        for step in (1, 1000, 20000):
            restored.postprocess_per_train_step(step, None, None, None, 128)
        torch.testing.assert_close(restored._means, means, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
