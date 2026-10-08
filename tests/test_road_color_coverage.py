import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from utils.lane_masks import image_lane_candidates, resize_lane_mask
from utils.road_supervision import (
    compute_road_supervision_losses, lane_edge_loss, masked_rgb_l1,
    road_supervision_metrics, supervision_weight, unpremultiply_road_rgb,
    valid_road_masks,
)
from datasets.qcraft.lane_supervision import attach_lane_masks


class RoadSupervisionTests(unittest.TestCase):
    @staticmethod
    def base_method(name):
        tree = ast.parse(Path('models/trainers/base.py').read_text(encoding='utf-8-sig'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BasicTrainer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        env = {'torch':torch, 'Dict':Dict, 'dataclass_gs':object, 'dataclass_camera':object}
        return method, env

    def test_render_color_override_keeps_all_depth_opacity_and_primary_info(self):
        method, env = self.base_method('render_gaussians')
        opacity = torch.tensor([[.7],[.8]],requires_grad=True)
        colors = torch.tensor([[.5,.6,.7],[.9,.9,.9]],requires_grad=True)
        marker = object()
        def rasterization(**kwargs):
            a = kwargs['opacities']
            weights = torch.stack([a[0],(1-a[0])*a[1]])
            rgb = (weights[:,None]*kwargs['colors']).sum(0)[None,None,None]
            return torch.cat([rgb,torch.ones_like(rgb[...,:1])],dim=-1),weights.sum().reshape(1,1,1,1),{'marker':marker}
        env['rasterization'] = rasterization
        exec(compile(ast.Module(body=[method],type_ignores=[]),'render_gaussians','exec'),env)
        self_obj = SimpleNamespace(training=False, render_cfg=SimpleNamespace(packed=False,absgrad=True,sparse_grad=False,antialiased=False,batch_size=1))
        gs = SimpleNamespace(means=torch.zeros((2,3)),quats=torch.ones((2,4)),scales=torch.ones((2,3)),opacities=opacity,rgbs=colors)
        cam = SimpleNamespace(camtoworlds=torch.eye(4),Ks=torch.eye(3),H=1,W=1)
        results, render = env['render_gaussians'](self_obj,gs,cam)
        original_info = self_obj.info
        tags = torch.tensor([[1.,0,0],[0,0,0]])
        contribution,_,total = render(colors_override=tags)
        _,_,road_alpha = render(torch.tensor([True,False]))
        self.assertAlmostEqual(contribution[0,0,0].item(),.7,places=6)
        self.assertTrue(torch.allclose(total,results['opacity']))
        self.assertAlmostEqual(road_alpha.item(),.7,places=6)
        self.assertIs(self_obj.info,original_info)
        self.assertIs(self_obj.info['marker'],marker)

    def test_real_affine_detach_keeps_road_gradients_without_updating_affine(self):
        method, env = self.base_method('affine_transformation')
        exec(compile(ast.Module(body=[method],type_ignores=[]),'affine_transformation','exec'),env)
        affine_bias = torch.tensor(.2,requires_grad=True)
        def affine(infos):
            return torch.cat([torch.eye(3),affine_bias.expand(3,1)],dim=1)
        self_obj = SimpleNamespace(models={'Affine':affine})
        rgb = torch.ones((2,2,3),requires_grad=True)
        result = env['affine_transformation'](self_obj,rgb,{},detach_affine=True)
        result.sum().backward()
        self.assertTrue((rgb.grad>0).all())
        self.assertIsNone(affine_bias.grad)

    def test_appearance_does_not_use_brightness_to_compensate_alpha(self):
        alpha = torch.tensor([[[0.3]]], requires_grad=True)
        color = torch.tensor([[[0.7, 0.6, 0.5]]], requires_grad=True)
        prediction = unpremultiply_road_rgb(alpha * color, alpha)
        masked_rgb_l1(prediction, torch.zeros_like(color), torch.ones((1, 1), dtype=torch.bool)).backward()
        self.assertLess(abs(alpha.grad.item()), 1e-6)
        self.assertTrue((color.grad > 0).all())

    def test_masks_exclude_occlusion_and_do_not_modify_inputs(self):
        road = torch.ones((9, 9), dtype=torch.bool)
        dynamic = torch.zeros_like(road)
        dynamic[4, 4] = True
        infos = {'road_masks': road, 'dynamic_masks': dynamic, 'lane_masks': road.clone()}
        valid, lane = valid_road_masks(infos, 1)
        self.assertFalse(valid[3:6, 3:6].any())
        self.assertFalse(valid[0].any())
        self.assertTrue(road.all())
        self.assertTrue(torch.equal(valid, lane))
        for key in ('egocar_masks', 'sky_masks', 'human_masks', 'vehicle_masks'):
            valid, _ = valid_road_masks({'road_masks': road, key: dynamic}, 0)
            self.assertFalse(valid[4, 4])

    def test_marking_coverage_is_not_diluted_and_zero_above_target(self):
        alpha = torch.ones((10, 10, 1), requires_grad=True)
        alpha.data[5, 5] = 0.2
        lane = torch.zeros((10, 10), dtype=torch.bool)
        lane[5, 5] = True
        cfg = {'road_coverage': {'w': .1, 'erosion_px': 0, 'lane_multiplier': 1}}
        infos = {'road_masks': torch.ones_like(lane), 'lane_masks': lane}
        losses = compute_road_supervision_losses({'RoadNodes_opacity': alpha}, infos, cfg, 0)
        self.assertAlmostEqual(losses['lane_coverage_loss'].item(), .1 * (.98-.2)**2, places=6)
        self.assertAlmostEqual(losses['road_coverage_loss'].item()*100, losses['lane_coverage_loss'].item(), places=6)
        sum(losses.values()).backward()
        self.assertLess(alpha.grad[5, 5].item(), 0)
        self.assertEqual(alpha.grad[0, 0].item(), 0)

    def test_class_contribution_penalizes_foreground_background_occlusion(self):
        # Isolated Road alpha can be high while a front Background dominates.
        road_alpha = torch.tensor([[[.99]]], requires_grad=True)
        bg_alpha = torch.tensor([[[.8]]], requires_grad=True)
        outputs = {'RoadNodes_opacity': road_alpha,
                   'opacity': bg_alpha + (1-bg_alpha)*road_alpha,
                   'RoadNodes_contribution': (1-bg_alpha)*road_alpha}
        losses = compute_road_supervision_losses(outputs, {'road_masks': torch.ones((1, 1), dtype=torch.bool)},
                   {'road_contribution': {'w': 1, 'erosion_px': 0}}, 0)
        losses['road_contribution_loss'].backward()
        self.assertGreater(bg_alpha.grad.item(), 0)
        self.assertLess(road_alpha.grad.item(), 0)

    def test_trainer_loss_entry_point_includes_color_coverage_and_share(self):
        method, env = self.base_method('compute_losses')
        env['Tensor'] = torch.Tensor
        exec(compile(ast.Module(body=[method],type_ignores=[]),'compute_losses','exec'),env)
        configs = OmegaConf.create({'rgb':{'w':.8},'ssim':{'w':.2},
            'road_color':{'w':.5,'erosion_px':0},
            'road_coverage':{'w':.1,'erosion_px':0},
            'road_contribution':{'w':.05,'erosion_px':0}})
        self_obj = SimpleNamespace(step=1000,losses_dict=configs,road_loss_fn=None,sky_opacity_loss_fn=None,
            depth_loss_fn=None,ssim=lambda a,b:torch.ones(()),models={},gaussian_classes={})
        color = torch.full((5,5,3),.3,requires_grad=True)
        alpha = torch.full((5,5,1),.5,requires_grad=True)
        share = torch.full((5,5,1),.25,requires_grad=True)
        outputs = {'rgb':color,'opacity':alpha,'RoadNodes_color':color,
                   'RoadNodes_opacity':alpha,'RoadNodes_contribution':share}
        lane = torch.zeros((5,5),dtype=torch.bool)
        lane[:,2] = True
        infos = {'pixels':torch.full_like(color,.6),'road_masks':torch.ones_like(lane),
                 'sky_masks':torch.zeros_like(lane),'lane_masks':lane}
        losses = env['compute_losses'](self_obj,outputs,infos,{},None)
        self.assertTrue({'road_color_loss','lane_color_loss','lane_edge_loss','road_coverage_loss',
                         'lane_coverage_loss','road_contribution_loss'} <= set(losses))
        sum(losses.values()).backward()
        self.assertTrue(torch.isfinite(color.grad).all())
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertTrue((share.grad < 0).all())

    def test_edge_loss_ignores_mask_boundaries_and_detects_blurred_marking(self):
        target = torch.zeros((3, 5, 3))
        target[:, 2] = 1
        lane = torch.zeros((3, 5), dtype=torch.bool)
        lane[:, 2] = True
        road = torch.ones_like(lane)
        self.assertEqual(lane_edge_loss(target, target, road, lane).item(), 0)
        self.assertGreater(lane_edge_loss(target*.5, target, road, lane).item(), 0)
        self.assertEqual(lane_edge_loss(target*.5, target, torch.zeros_like(road), lane).item(), 0)

    def test_empty_masks_disabled_configs_and_metrics(self):
        infos = {'road_masks': torch.zeros((3, 3), dtype=torch.bool)}
        outputs = {'RoadNodes_opacity': torch.ones((3, 3, 1))}
        self.assertEqual(compute_road_supervision_losses(outputs, infos, {'road_color': {'w': 1}}, 0), {})
        self.assertEqual(road_supervision_metrics(outputs, infos)['road_supervised_pixels'].item(), 0)
        self.assertEqual(supervision_weight({'w': 1, 'start_after': 500}, 499), 0)
        self.assertAlmostEqual(supervision_weight({'w': 1, 'start_after': 0, 'warmup_steps': 1000}, 0), .001)
        self.assertEqual(supervision_weight({'w': 1, 'warmup_steps': 1000}, 1000), 1)

    def test_render_supervision_branch_uses_independent_rgb_and_class_tags(self):
        # Exercise the actual forward block without importing CUDA dependencies.
        tree = ast.parse(Path('models/trainers/scene_graph.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        start = next(i for i,n in enumerate(forward.body) if isinstance(n, ast.ImportFrom) and n.module == 'utils.road_supervision')
        end = next(i for i in range(start+1, len(forward.body)) if isinstance(forward.body[i], ast.If)) + 1
        block = compile(ast.Module(body=forward.body[start:end], type_ignores=[]), 'road_forward_block', 'exec')
        color = torch.tensor([[.5,.6,.7],[.9,.9,.9]], requires_grad=True)
        alpha = torch.tensor([.7,.8], requires_grad=True)
        affine_bias = torch.tensor(.1, requires_grad=True)
        def render(mask=None, colors_override=None):
            a = alpha if mask is None else alpha * mask
            weights = torch.stack([a[0], (1-a[0])*a[1]])
            colors = color if colors_override is None else colors_override
            return (weights[:,None]*colors).sum(0)[None,None], None, weights.sum()[None,None,None]
        self_obj = SimpleNamespace(training=True, step=1000,
            losses_dict={'road_color': {'w': 1}, 'road_coverage': {'w': 1}, 'road_contribution': {'w': 1}},
            gaussian_classes={'RoadNodes': 5}, pts_labels=torch.tensor([5,0]),
            affine_transformation=lambda rgb, infos, detach_affine=False: rgb + (affine_bias.detach() if detach_affine else affine_bias))
        env = {'self': self_obj, 'torch': torch, 'gs': SimpleNamespace(rgbs=color), 'image_infos': {'road_masks':torch.ones((1,1),dtype=torch.bool)}, 'outputs': {}, 'render_fn':render}
        exec(block, env)
        outputs = env['outputs']
        self.assertTrue(torch.allclose(outputs['RoadNodes_color'][0,0], color[0]+.1))
        self.assertAlmostEqual(outputs['RoadNodes_contribution'].item(), .7, places=6)
        outputs['RoadNodes_color'].sum().backward()
        self.assertTrue(torch.equal(color.grad[1], torch.zeros(3)))
        self.assertIsNone(affine_bias.grad)
        self.assertLess(alpha.grad.abs().max().item(), 1e-6)
        self_obj.training = False
        env['outputs'] = {}
        exec(block, env)
        self.assertEqual(env['outputs'], {})


class LaneMaskTests(unittest.TestCase):
    def test_white_yellow_markings_and_no_uniform_bright_region(self):
        rgb = torch.full((31,31,3), .25)
        rgb[:, 10] = .9
        rgb[:, 20] = torch.tensor([.9,.8,.1])
        road = torch.ones((31,31), dtype=torch.bool)
        lane = image_lane_candidates(rgb, road)
        self.assertTrue(lane[:,10].all())
        self.assertTrue(lane[:,20].all())
        self.assertFalse(lane[:,0].any())
        self.assertFalse(image_lane_candidates(torch.full_like(rgb,.9),road).any())
        self.assertFalse(image_lane_candidates(rgb,torch.zeros_like(road)).any())
        thin = torch.zeros((8,8), dtype=torch.bool)
        thin[:,1] = True
        self.assertTrue(resize_lane_mask(thin,(2,2))[:,0].all())

    def test_explicit_labels_take_priority_and_dynamic_regions_are_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'lane_masks').mkdir()
            label = np.zeros((5,5),dtype=np.uint8)
            label[:,2] = 255
            Image.fromarray(label).save(root/'lane_masks/000010_2.png')
            dynamic = torch.zeros((1,5,5),dtype=torch.bool)
            dynamic[0,2,2] = True
            camera = SimpleNamespace(data_path=tmp, start_timestep=10, cam_id=2, num_frames=1,
                WIDTH=5, HEIGHT=5, images=torch.zeros((1,5,5,3)), road_masks=torch.ones((1,5,5),dtype=torch.bool),
                egocar_mask=torch.zeros((5,5),dtype=torch.bool), dynamic_masks=dynamic, undistort=False)
            attach_lane_masks({2:camera}, {'lane_mask_source':'auto'})
            self.assertEqual(camera.lane_masks.sum().item(),4)
            self.assertFalse(camera.lane_masks[0,2,2])
            with self.assertRaises(FileNotFoundError):
                attach_lane_masks({2:camera}, {'lane_mask_source':'labels', 'lane_mask_dir':str(root/'missing')})


if __name__ == '__main__':
    unittest.main()
