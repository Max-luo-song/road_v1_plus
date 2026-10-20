"""CPU tests of real RoadNodes methods, without importing CUDA-only dependencies."""
import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple
import unittest

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.nn import Parameter

from utils.opacity import bounded_logit, bounded_sigmoid
from utils.road_scales import decode_road_scales, encode_road_scales


def road_class():
    tree = ast.parse(Path('models/nodes/road.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RoadNodes')
    env = dict(torch=torch, nn=nn, Parameter=Parameter, OmegaConf=OmegaConf,
               Dict=Dict, List=List, Tuple=Tuple, math=math, logger=logging.getLogger(),
               bounded_logit=bounded_logit, bounded_sigmoid=bounded_sigmoid,
               decode_road_scales=decode_road_scales, encode_road_scales=encode_road_scales,
               num_sh_bases=lambda degree: (degree + 1) ** 2,
               RGB2SH=lambda rgb: (rgb - .5) / .28209479177387814,
               dataclass_camera=object)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'road.py', 'exec'), env)
    return env['RoadNodes']


class AdaptiveFootprintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Road = road_class()

    def model(self, smooth=True, alpha=.01, lower=.01):
        return self.Road('RoadNodes', ctrl=OmegaConf.create({
            'road_sh_degree': 0, 'smooth_xy_scale': smooth,
            'scale_xy_min': lower, 'scale_xy_max': .08}),
            reg=OmegaConf.create({'alpha_min': alpha, 'lowest_opacity': .01,
                                  'legacy_opacity_floor': .2}),
            init={'opacity_init_value': .7}, device=torch.device('cpu'))

    @staticmethod
    def checkpoint(marked=False):
        state = {
            '_means': torch.tensor([[0., 0., 0.], [.05, 0., 0.]]),
            '_scales': torch.tensor([[.04, .01, .005], [.1, .03, .005]]).log(),
            '_quats': torch.tensor([[1., 0, 0, 0], [1., 0, 0, 0]]),
            '_opacities': torch.tensor([[.1], [.7]]).logit(),
            '_features_dc': torch.zeros(2, 3),
            '_features_rest': torch.zeros(2, 0, 3),
        }
        if marked:
            state['_opacity_alpha_min'] = torch.tensor(.2, dtype=torch.float64)
        return state

    def test_initialization_preserves_physical_scale_and_alpha(self):
        model = self.model()
        points = torch.tensor([[0., 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, 0]])
        self.assertFalse(model.create_from_pcd(points, torch.full_like(points, .5)))
        expected = torch.tensor([.04, .04, .005]).expand(model.num_points, -1)
        torch.testing.assert_close(model.get_scaling, expected)
        torch.testing.assert_close(model.get_opacity, torch.full((model.num_points, 1), .7))

    def test_legacy_checkpoints_preserve_rendered_scales_and_opacity(self):
        for marked in (False, True):
            state = self.checkpoint(marked)
            expected_scale = decode_road_scales(state['_scales'], (-1, -1))
            raw_alpha = state['_opacities']
            expected_alpha = bounded_sigmoid(raw_alpha, .2) if marked else raw_alpha.sigmoid().clamp_min(.2)
            model = self.model()
            model.load_state_dict(state, strict=True)
            torch.testing.assert_close(model.get_scaling, expected_scale, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(model.get_opacity, expected_alpha)
            self.assertNotIn('_scale_xy_bounds', state)  # caller's dict is untouched

    def test_roundtrip_and_changed_bounds_preserve_physical_values(self):
        original = self.model()
        original.load_state_dict(self.checkpoint(True))
        for lower in (.01, .005):
            restored = self.model(lower=lower)
            restored.load_state_dict(original.state_dict(), strict=True)
            torch.testing.assert_close(restored.get_scaling, original.get_scaling)
            torch.testing.assert_close(restored.get_opacity, original.get_opacity)
            torch.testing.assert_close(restored._means, original._means)
            if lower == .01:
                torch.testing.assert_close(restored._scales, original._scales, rtol=0, atol=0)

    def test_legacy_mode_remains_available(self):
        smooth = self.model()
        smooth.load_state_dict(self.checkpoint(True))
        legacy = self.model(smooth=False, alpha=.2)
        legacy.load_state_dict(smooth.state_dict(), strict=True)
        torch.testing.assert_close(legacy.get_scaling, smooth.get_scaling)
        torch.testing.assert_close(legacy.get_opacity, smooth.get_opacity)

    def test_scale_and_opacity_can_optimize_below_old_floors(self):
        model = self.model()
        model.load_state_dict(self.checkpoint(True))
        # Optimize physical properties, rather than checking the encoding formula alone.
        optim = torch.optim.Adam([model._scales, model._opacities], lr=.1)
        for _ in range(180):
            optim.zero_grad()
            loss = ((model.get_scaling[:, 0] - .012) / .04).square().mean()
            loss = loss + (model.get_opacity - .03).square().mean()
            loss.backward()
            optim.step()
        self.assertTrue((model.get_scaling[:, 0] < .02).all())
        self.assertTrue((model.get_scaling[:, 0] > .01).all())
        self.assertTrue((model.get_opacity < .2).all())
        self.assertTrue((model.get_opacity > .01).all())

    def test_actual_trainer_hooks_freeze_position_rotation_and_thickness(self):
        model = self.model()
        model.load_state_dict(self.checkpoint(True))
        gs = SimpleNamespace(**{name: value for name, value in {
            'means': model._means, 'quats': model.get_quats,
            'scales': model.get_scaling, 'opacities': model.get_opacity}.items()})
        source = Path('models/trainers/scene_graph.py').read_text()
        tree = ast.parse(source)
        block = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Attribute) and n.test.attr == 'training'
                     and 'gs.means.requires_grad_' in ast.get_source_segment(source, n))
        env = dict(self=SimpleNamespace(training=True), gs=gs, freeze_road=True,
                   freeze_mask=torch.ones(2, dtype=torch.bool))
        exec(compile(ast.Module(body=[block], type_ignores=[]), 'freeze_hooks', 'exec'), env)
        (gs.means.sum() + gs.quats.sum() + gs.scales.sum() + gs.opacities.sum()).backward()
        self.assertTrue((model._means.grad == 0).all())
        self.assertTrue((model._quats.grad == 0).all())
        self.assertTrue((model._scales.grad[:, 2] == 0).all())
        self.assertTrue((model._scales.grad[:, :2] > 0).all())
        self.assertTrue((model._opacities.grad > 0).all())

    def test_default_postprocess_does_not_add_or_move_points(self):
        model = self.model()
        model.load_state_dict(self.checkpoint(True))
        means = model._means.detach().clone()
        for step in (0, 1000, 20000, 40000):
            model.postprocess_per_train_step(step, None, None, None, 1024)
        torch.testing.assert_close(model._means, means, rtol=0, atol=0)

    def test_bounds_are_validated(self):
        with self.assertRaises(ValueError):
            self.model(lower=.08)


if __name__ == '__main__':
    unittest.main()
