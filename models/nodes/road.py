from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import logging

import torch
import torch.nn as nn
from torch.nn import Parameter
from pytorch3d.ops import knn_points

from models.gaussians.basics import *
from utils.opacity import bounded_logit, bounded_sigmoid
from utils.road_scales import decode_road_scales, encode_road_scales

logger = logging.getLogger()

class RoadNodes(nn.Module):

    def __init__(
        self,
        class_name: str,
        ctrl: OmegaConf,
        reg: OmegaConf = None,
        networks: OmegaConf = None,
        scene_scale: float = 30.,
        scene_origin: torch.Tensor = torch.zeros(3),
        num_train_images: int = 300,
        device: torch.device = torch.device("cuda"),
        **kwargs
    ):
        super().__init__()
        self.class_prefix = class_name + "#"
        self.ctrl_cfg = ctrl
        self.reg_cfg = reg
        self.init_cfg = kwargs.get("init", {})
        # -1 identifies the original sigmoid + clamp parameterization.
        alpha_min = float(self.reg_cfg.get("alpha_min", -1.0))
        if alpha_min != -1.0 and not 0.0 <= alpha_min < 1.0:
            raise ValueError("alpha_min must be in [0, 1)")
        self.register_buffer(
            "_opacity_alpha_min", torch.tensor(alpha_min, dtype=torch.float64, device=device)
        )
        self.networks_cfg = networks
        self.scene_scale = scene_scale
        self.scene_origin = scene_origin
        self.num_train_images = num_train_images
        self.step = 0
        
        self.device = device
        self.ball_gaussians=self.ctrl_cfg.get("ball_gaussians", False)
        self.gaussian_2d = self.ctrl_cfg.get("gaussian_2d", False)
        smooth_xy = self.ctrl_cfg.get("smooth_xy_scale", False)
        bounds = (-1., -1.)
        if smooth_xy:
            if self.ball_gaussians or self.gaussian_2d:
                raise ValueError("smooth_xy_scale requires ordinary 3D RoadNodes")
            bounds = (float(self.ctrl_cfg.get("scale_xy_min", .01)),
                      float(self.ctrl_cfg.get("scale_xy_max", .08)))
            if not 0 < bounds[0] < bounds[1]:
                raise ValueError("Road XY scale bounds must satisfy 0 < min < max")
        self.register_buffer("_scale_xy_bounds", torch.tensor(
            bounds, dtype=torch.float64, device=device))
        self._scale_xy_limits = bounds
        
        # for evaluation
        self.in_test_set = False
        
        # init models
        self.xys_grad_norm = None
        self.max_2Dsize = None
        self._means = torch.zeros(1, 3, device=self.device)
        if self.ball_gaussians:
            self._scales = torch.zeros(1, 1, device=self.device)
        else:
            if self.gaussian_2d:
                self._scales = torch.zeros(1, 2, device=self.device)
            else:
                self._scales = torch.zeros(1, 3, device=self.device)
        self._quats = torch.zeros(1, 4, device=self.device)
        self._opacities = torch.zeros(1, 1, device=self.device)
        self._features_dc = torch.zeros(1, 3, device=self.device)
        self._features_rest = torch.zeros(1, num_sh_bases(self.sh_degree) - 1, 3, device=self.device)
        self._lane_mask = None
        self.register_buffer("_hf_detail_mask", torch.zeros(1, dtype=torch.bool, device=device))
        
    @property
    def sh_degree(self):
        return self.ctrl_cfg.road_sh_degree

    def create_from_pcd(
        self,
        init_means: torch.Tensor,
        init_colors: torch.Tensor,
        lane_samples: Dict[str, torch.Tensor] = None,
        road_cameras=None,
        road_train_frames=None,
        initialization_output_dir=None,
    ) -> bool:
        if init_means.numel() == 0 or init_colors.numel() == 0:
            logger.warning(f"RoadNodes initialization skipped: empty input point cloud for {self.class_prefix}")
            return True

        grid_spacing = 0.1 # 10cm一个点
        k = 3
        chunk_size = 10000  # 每次处理1万个网格点，显存压力极小

        # 1. 建立基础网格(以grid_spacing为基准，再偏移一些)
        x_min, x_max = init_means[:, 0].min(), init_means[:, 0].max()
        y_min, y_max = init_means[:, 1].min(), init_means[:, 1].max()
        if torch.isclose(x_min, x_max) or torch.isclose(y_min, y_max):
            logger.warning(f"RoadNodes initialization skipped: insufficient spatial extent for {self.class_prefix}")
            return True

        x_coords = torch.arange(x_min, x_max, grid_spacing, device=self.device)
        y_coords = torch.arange(y_min, y_max, grid_spacing, device=self.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        grid_a = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        offset = grid_spacing / 2.0
        grid_b = grid_a + torch.tensor([offset, 0.0], device=self.device)   # X 偏移
        grid_c = grid_a + torch.tensor([0.0, offset], device=self.device)   # Y 偏移
        grid_d = grid_a + torch.tensor([offset, offset], device=self.device) # 对角线偏移
        grid_xy_all = torch.cat([grid_a, grid_b, grid_c, grid_d], dim=0)

        resampled_means_list = []
        resampled_colors_list = []

        # 2. 分块处理网格点
        for i in range(0, grid_xy_all.shape[0], chunk_size):
            grid_xy = grid_xy_all[i : i + chunk_size]
            
            # 只计算当前 chunk 到原始点云的距离
            dist_mat = torch.cdist(grid_xy, init_means[:, :2]) 
            topk_dist, topk_idx = torch.topk(dist_mat, k, largest=False, dim=-1)
            del dist_mat
            torch.cuda.empty_cache()
            
            valid_mask = topk_dist[:, 0] < (grid_spacing * 4.0) #阈值调大一些
            if not valid_mask.any(): continue

            # 执行插值逻辑 
            curr_weights = 1.0 / (topk_dist[valid_mask] + 1e-6)
            curr_weights /= curr_weights.sum(dim=-1, keepdim=True)
            
            interp_z = (init_means[topk_idx[valid_mask], 2] * curr_weights).sum(dim=-1, keepdim=True)
            resampled_means_list.append(torch.cat([grid_xy[valid_mask], interp_z], dim=-1))
            
            neighbor_rgb = init_colors[topk_idx[valid_mask]]
            resampled_colors_list.append((neighbor_rgb * curr_weights[..., None]).sum(dim=1))

        if len(resampled_means_list) == 0:
            logger.warning(f"RoadNodes initialization skipped: no valid resampled points generated for {self.class_prefix}")
            return True

        resampled_means = torch.cat(resampled_means_list, dim=0)
        resampled_colors = torch.cat(resampled_colors_list, dim=0)

        base_count = len(resampled_means)
        hf_cfg = self.init_cfg.get("high_frequency", {})
        if hf_cfg.get("enabled", False):
            if self.ball_gaussians or self.gaussian_2d or not self.ctrl_cfg.get("smooth_xy_scale", False):
                raise ValueError("HF road initialization requires ordinary 3D RoadNodes with smooth_xy_scale")
            if self.ctrl_cfg.get("edge_split_start") is not None or self.ctrl_cfg.get("edge_split_step") is not None:
                raise ValueError("HF fixed-grid initialization requires training-time road splitting disabled")
            from utils.road_high_frequency import initialize_high_frequency_road
            extra_means, extra_colors, self.hf_initialization_stats = initialize_high_frequency_road(
                resampled_means, resampled_colors, road_cameras, road_train_frames,
                hf_cfg, output_dir=initialization_output_dir)
            resampled_means = torch.cat((resampled_means, extra_means))
            resampled_colors = torch.cat((resampled_colors, extra_colors))
        self._hf_detail_mask = torch.arange(len(resampled_means), device=self.device) >= base_count

        self._means = Parameter(resampled_means)
        num_points = resampled_means.shape[0]
        
        fixed_value = grid_spacing * 0.4
        fixed_scale_tensor = torch.full((num_points, 1), fixed_value, device=self.device)
        epsilon = 0.005
        # NOTE(gls)：让高斯在 XY 方向平铺，Z 方向极薄
        base_scale = torch.cat([fixed_scale_tensor, fixed_scale_tensor, torch.ones_like(fixed_scale_tensor) * epsilon], dim=-1)

        if lane_samples is not None:
            lane_points = lane_samples["points"]
            lane_tangents = lane_samples["tangents"]
            lane_dist2, lane_indices, _ = knn_points(
                resampled_means[None, :, :2], lane_points[None, :, :2], K=1
            )
            lane_indices = lane_indices[0, :, 0]
            lane_dist = torch.sqrt(lane_dist2[0, :, 0].clamp_min(0.0))
            lane_height_error = torch.abs(
                resampled_means[:, 2] - lane_points[lane_indices, 2]
            )
            self._lane_mask = (
                (lane_dist <= self.ctrl_cfg.get("lane_influence_radius", 0.2))
                & (lane_height_error <= self.ctrl_cfg.get("lane_height_tolerance", 0.3))
            )
            base_scale[self._lane_mask, 0] = self.ctrl_cfg.get(
                "lane_scale_parallel", 0.04
            )
            base_scale[self._lane_mask, 1] = self.ctrl_cfg.get(
                "lane_scale_perpendicular", 0.02
            )

        if self._hf_detail_mask.any():
            detail_scale = float(hf_cfg.get("detail_scale", .015))
            if not self._scale_xy_limits[0] < detail_scale < self._scale_xy_limits[1]:
                raise ValueError("HF detail_scale must be strictly inside XY scale bounds")
            base_scale[self._hf_detail_mask, :2] = detail_scale

        if self.ball_gaussians:
            # 如果是球形，取最大值
            self._scales = Parameter(torch.log(fixed_scale_tensor))
        else:
            self._scales = Parameter(encode_road_scales(base_scale, self._scale_xy_limits))

        # 旋转初始化：w=1, 意味着高斯主轴对齐世界坐标轴 (配合你压缩 Z 的策略，正好贴合路面)
        unit_quats = torch.zeros((num_points, 4), dtype=torch.float, device=self.device)
        unit_quats[:, 0] = 1.0 
        if self._lane_mask is not None and self._lane_mask.any():
            tangents = lane_tangents[lane_indices[self._lane_mask]]
            yaw = torch.atan2(tangents[:, 1], tangents[:, 0])
            unit_quats[self._lane_mask, 0] = torch.cos(yaw * 0.5)
            unit_quats[self._lane_mask, 3] = torch.sin(yaw * 0.5)
            logger.info(
                f"Lane-aware RoadNodes: {self._lane_mask.sum().item()}/{num_points} "
                f"Gaussians within {self.ctrl_cfg.get('lane_influence_radius', 0.2):.2f} m"
            )
        self._quats = Parameter(unit_quats)

        if self._lane_mask is not None and self._lane_mask.any():
            lane_mask = self._lane_mask

            def freeze_lane_perpendicular_scale(grad):
                grad = grad.clone()
                grad[lane_mask, 1] = 0
                return grad

            self._scales.register_hook(freeze_lane_perpendicular_scale)

        # SH 颜色初始化
        dim_sh = num_sh_bases(self.sh_degree)
        fused_color = RGB2SH(resampled_colors) 
        shs = torch.zeros((num_points, dim_sh, 3), device=self.device)
        
        if self.sh_degree > 0:
            shs[:, 0, :3] = fused_color
            # 高阶系数置零
            shs[:, 1:, :] = 0.0
        else:
            shs[:, 0, :3] = torch.logit(resampled_colors, eps=1e-10)
            
        self._features_dc = Parameter(shs[:, 0, :])
        self._features_rest = Parameter(shs[:, 1:, :])
        
        # 透明度初始化(会训练，初始化随机即可)
        if self.reg_cfg.get("alpha_min", None) is not None:
            opacity_init_value = float(self.init_cfg.get("opacity_init_value", 0.7))
            alpha_min = float(self.reg_cfg["alpha_min"])
            if not alpha_min < opacity_init_value < 1.0:
                raise ValueError("opacity_init_value must be between alpha_min and 1")
            self._opacities = Parameter(self._opacity_to_raw(
                torch.full((num_points, 1), opacity_init_value, device=self.device)
            ))
        else:
            self._opacities = Parameter(torch.logit(0.2 * torch.ones(num_points, 1, device=self.device)))
        if self._hf_detail_mask.any():
            detail_opacity = float(hf_cfg.get("detail_opacity", .1))
            alpha_floor = float(self.reg_cfg.get("alpha_min", self.reg_cfg.get("lowest_opacity", 0)))
            if not alpha_floor < detail_opacity < 1:
                raise ValueError("HF detail_opacity must exceed the configured opacity floor")
            with torch.no_grad():
                self._opacities[self._hf_detail_mask] = self._opacity_to_raw(
                    self._opacities.new_full((int(self._hf_detail_mask.sum()), 1), detail_opacity))
        self._register_hf_gradient_hooks()
        return False

    def _register_hf_gradient_hooks(self):
        """Progressively unlock detail points, including under Adam normalization."""
        warmup = int(self.init_cfg.get("high_frequency", {}).get("detail_warmup_steps", 0))
        if warmup <= 0 or not self._hf_detail_mask.any():
            return

        # A scalar gradient multiplier is largely canceled by Adam. Instead,
        # unlock a growing subset of detail points with exact zero gradients
        # for those not yet active. Base points are always active.
        activation = torch.zeros_like(self._hf_detail_mask, dtype=torch.long)
        count = int(self._hf_detail_mask.sum())
        activation[self._hf_detail_mask] = torch.ceil(
            torch.arange(1, count + 1, device=self.device) * (warmup / count)).long()

        def detail_gradient(grad):
            if self.step + 1 >= warmup:
                return grad
            multiplier = (activation <= self.step + 1).to(grad)
            return grad * multiplier.reshape([-1] + [1] * (grad.ndim - 1))

        for parameter in (self._features_dc, self._features_rest, self._scales, self._opacities):
            parameter.register_hook(detail_gradient)
    
    @property
    def colors(self):
        if self.sh_degree > 0:
            return SH2RGB(self._features_dc)
        else:
            return torch.sigmoid(self._features_dc)
    @property
    def shs_0(self):
        return self._features_dc
    @property
    def shs_rest(self):
        return self._features_rest
    @property
    def num_points(self):
        return self._means.shape[0]
    @property
    def get_scaling(self):
        if self.ball_gaussians:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales).repeat(1, 2)
                scaling = torch.cat([scaling, torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return torch.exp(self._scales).repeat(1, 3)
        else:
            if self.gaussian_2d:
                scaling = torch.exp(self._scales)
                scaling = torch.cat([scaling[..., :2], torch.zeros_like(scaling[..., :1])], dim=-1)
                return scaling
            else:
                return decode_road_scales(self._scales, self._scale_xy_limits)
    @property
    def get_opacity(self):
        alpha_min = self.reg_cfg.get("alpha_min", None)
        if alpha_min is not None:
            return bounded_sigmoid(self._opacities, float(alpha_min))
        lowest_opacity = self.reg_cfg.get("lowest_opacity", None)
        # print("lowest_opacity:", lowest_opacity)
        return torch.sigmoid(self._opacities).clamp(min=lowest_opacity)

    def _opacity_to_raw(self, opacity: torch.Tensor) -> torch.Tensor:
        alpha_min = self.reg_cfg.get("alpha_min", None)
        if alpha_min is not None:
            return bounded_logit(opacity, float(alpha_min))
        return torch.logit(opacity)

    @property
    def get_quats(self):
        return self.quat_act(self._quats)
    
    # NOTE(gls)：更新加一个epsilon，防止除0
    def quat_act(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True)
        # 防止范数为 0（添加一个很小的 epsilon，如 1e-8）
        norm = torch.clamp(norm, min=1e-8)
        return x / norm

    def preprocess_per_train_step(self, step: int):
        self.step = step
    
    # NOTE(gls): 关闭路面致密化
    def postprocess_per_train_step(
        self,
        step: int,
        optimizer: torch.optim.Optimizer,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        split_start = self.ctrl_cfg.get("edge_split_start", None)
        split_step = self.ctrl_cfg.get("edge_split_step", None)
        if split_start is None or split_step is None:
            return
        if split_start is not None and step == split_start:
            self.xys_grad_norm = None
            self.vis_counts = None
            self.max_2Dsize = None

        self.after_train(radii, xys_grad, last_size)
        if split_step is not None and step == split_step:
            self.edge_split(optimizer)

    def edge_split(self, optimizer: torch.optim.Optimizer) -> None:
        ratio = self.ctrl_cfg.get("edge_split_ratio", 0.05)
        scores = self.xys_grad_norm / self.vis_counts.clamp_min(1)
        valid = scores > 0
        if not valid.any():
            logger.warning("Road edge split skipped: no visible road Gaussians")
            return
        n_split = min(valid.sum().item(), max(1, int(valid.sum().item() * ratio)))
        split_mask = torch.zeros_like(valid)
        split_mask[torch.topk(scores.masked_fill(~valid, -1), n_split).indices] = True

        parent_means = self._means.detach()[split_mask]
        parent_scales = self.get_scaling.detach()[split_mask]
        offsets = parent_scales[:, None, :2] * parent_means.new_tensor(
            [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
        )[None]
        child_means = parent_means[:, None].repeat(1, 4, 1)
        child_means[:, :, :2] += offsets

        if self._scale_xy_limits[0] > 0:
            child_scales = parent_scales[:, None].repeat(1, 4, 1)
            child_scales[:, :, :2] /= 2.0
            child_scales = encode_road_scales(child_scales, self._scale_xy_limits)
        else:
            child_scales = self._scales.detach()[split_mask, None].repeat(1, 4, 1)
            child_scales[:, :, :2] -= math.log(2.0)

        self._means.data[split_mask] = child_means[:, 0]
        self._scales.data[split_mask] = child_scales[:, 0]
        self._means = Parameter(torch.cat([self._means.detach(), child_means[:, 1:].reshape(-1, 3)], dim=0))
        self._scales = Parameter(torch.cat([self._scales.detach(), child_scales[:, 1:].reshape(-1, 3)], dim=0))
        self._quats = Parameter(torch.cat([self._quats.detach(), self._quats.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))
        self._features_dc = Parameter(torch.cat([self._features_dc.detach(), self._features_dc.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))
        self._features_rest = Parameter(torch.cat([self._features_rest.detach(), self._features_rest.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))
        self._opacities = Parameter(torch.cat([self._opacities.detach(), self._opacities.detach()[split_mask].repeat_interleave(3, dim=0)], dim=0))

        dup_in_optim(optimizer, split_mask, self.get_gaussian_param_groups(), n=3)
        logger.info(f"Road edge split: {n_split} parents -> {n_split * 4} children")
        self.xys_grad_norm = None
        self.vis_counts = None
        self.max_2Dsize = None

    def after_train(
        self,
        radii: torch.Tensor,
        xys_grad: torch.Tensor,
        last_size: int,
    ) -> None:
        with torch.no_grad():
            n_filtered = int(self.filter_mask.sum().item())
            radii = reduce_per_gaussian(radii, n_filtered, reduce="amax")
            grads = reduce_per_gaussian(xys_grad.norm(dim=-1), n_filtered, reduce="mean")
            
            # keep track of a moving average of grad norms
            visible_mask = radii > 0
            full_mask = torch.zeros(self.num_points, device=radii.device, dtype=torch.bool)
            full_mask[self.filter_mask] = visible_mask
            
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(self.num_points, device=grads.device, dtype=grads.dtype)
                self.xys_grad_norm[self.filter_mask] = grads
                self.vis_counts = torch.ones_like(self.xys_grad_norm)
            else:
                assert self.vis_counts is not None
                self.vis_counts[full_mask] = self.vis_counts[full_mask] + 1
                self.xys_grad_norm[full_mask] = grads[visible_mask] + self.xys_grad_norm[full_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros(self.num_points, device=radii.device, dtype=torch.float32)
            newradii = radii[visible_mask]
            self.max_2Dsize[full_mask] = torch.maximum(
                self.max_2Dsize[full_mask], newradii / float(last_size)
            )
        
    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {
            self.class_prefix+"xyz": [self._means],
            self.class_prefix+"sh_dc": [self._features_dc],
            self.class_prefix+"sh_rest": [self._features_rest],
            self.class_prefix+"opacity": [self._opacities],
            self.class_prefix+"scaling": [self._scales],
            self.class_prefix+"rotation": [self._quats],
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.get_gaussian_param_groups()

    # NOTE(gls)： 受控致密化，体现在致密化高斯限制在路面上
    def refinement_after_undercontrol(self, step, optimizer: torch.optim.Optimizer) -> None:
            assert step == self.step
            if self.step <= self.ctrl_cfg.warmup_steps:
                return
                
            # 1. 定义路面厚度约束
            ROAD_EPSILON = 0.005
            ROAD_LOG_EPSILON = math.log(ROAD_EPSILON)

            with torch.no_grad():
                reset_interval = self.ctrl_cfg.reset_alpha_interval
                # 开启致密化
                do_densification = (
                    self.step < self.ctrl_cfg.stop_split_at
                    and self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval)
                )
                
                print(f"Class {self.class_prefix} current points: {self.num_points} @ step {self.step}")
                
                if do_densification:
                    assert self.xys_grad_norm is not None and self.vis_counts is not None and self.max_2Dsize is not None
                    
                    avg_grad_norm = self.xys_grad_norm / self.vis_counts
                    high_grads = (avg_grad_norm > self.ctrl_cfg.densify_grad_thresh).squeeze()
                    
                    # 计算当前路面的平均 Z 轴高度，确保新点不会偏移
                    avg_road_z = self._means[:, 2].mean()

                    # --- Split 逻辑 ---
                    splits = (self.get_scaling.max(dim=-1).values > self.ctrl_cfg.densify_size_thresh * self.scene_scale).squeeze()
                    if self.step < self.ctrl_cfg.stop_screen_size_at:
                        splits |= (self.max_2Dsize > self.ctrl_cfg.split_screen_size).squeeze()
                    splits &= high_grads
                    nsamps = self.ctrl_cfg.n_split_samples
                    (split_means, split_feature_dc, split_feature_rest, split_opacities, split_scales, split_quats) = self.split_gaussians(splits, nsamps)

                    # 投影 Split 生成的点：强制 Z 轴位置和 Scale
                    split_means[:, 2] = avg_road_z
                    split_scales[:, 2] = ROAD_LOG_EPSILON

                    # --- Duplicate 逻辑 ---
                    dups = (self.get_scaling.max(dim=-1).values <= self.ctrl_cfg.densify_size_thresh * self.scene_scale).squeeze()
                    dups &= high_grads
                    (dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats) = self.dup_gaussians(dups)
                    
                    # 投影 Duplicate 生成的点：强制 Z 轴位置和 Scale
                    dup_means[:, 2] = avg_road_z
                    dup_scales[:, 2] = ROAD_LOG_EPSILON

                    # 合并参数
                    self._means = Parameter(torch.cat([self._means.detach(), split_means, dup_means], dim=0))
                    self._features_dc = Parameter(torch.cat([self._features_dc.detach(), split_feature_dc, dup_feature_dc], dim=0))
                    self._features_rest = Parameter(torch.cat([self._features_rest.detach(), split_feature_rest, dup_feature_rest], dim=0))
                    self._opacities = Parameter(torch.cat([self._opacities.detach(), split_opacities, dup_opacities], dim=0))
                    self._scales = Parameter(torch.cat([self._scales.detach(), split_scales, dup_scales], dim=0))
                    self._quats = Parameter(torch.cat([self._quats.detach(), split_quats, dup_quats], dim=0))
                    
                    self.max_2Dsize = torch.cat([self.max_2Dsize, torch.zeros_like(split_scales[:, 0]), torch.zeros_like(dup_scales[:, 0])], dim=0)
                    
                    param_groups = self.get_gaussian_param_groups()
                    dup_in_optim(optimizer, torch.where(splits)[0], param_groups, n=nsamps)
                    dup_in_optim(optimizer, torch.where(dups)[0], param_groups, 1)

                # --- Cull & Reset 逻辑 (保持原样) ---
                if self.step % reset_interval > max(self.num_train_images, self.ctrl_cfg.refine_interval):
                    deleted_mask = self.cull_gaussians()
                    param_groups = self.get_gaussian_param_groups()
                    remove_from_optim(optimizer, deleted_mask, param_groups)
                
                print(f"Class {self.class_prefix} left points: {self.num_points}")
                        
                if self.step % reset_interval == self.ctrl_cfg.refine_interval:
                    reset_value = torch.min(self.get_opacity.data, torch.ones_like(self._opacities.data) * self.ctrl_cfg.reset_alpha_value)
                    self._opacities.data = self._opacity_to_raw(reset_value)
                    for group in optimizer.param_groups:
                        if group["name"] == self.class_prefix+"opacity":
                            param_state = optimizer.state[group["params"][0]]
                            param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                            param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])

                self.xys_grad_norm = None
                self.vis_counts = None
                self.max_2Dsize = None

    def cull_gaussians(self):
        """
        This function deletes gaussians with under a certain opacity threshold
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacity.data < self.ctrl_cfg.cull_alpha_thresh).squeeze()
        if self.step > self.ctrl_cfg.reset_alpha_interval:
            # cull huge ones
            physical_scales = self.get_scaling if self._scale_xy_limits[0] > 0 else self._scales.exp()
            toobigs = (
                physical_scales.max(dim=-1).values >
                self.ctrl_cfg.cull_scale_thresh * self.scene_scale
            ).squeeze()
            culls = culls | toobigs
            if self.step < self.ctrl_cfg.stop_screen_size_at:
                # cull big screen space
                assert self.max_2Dsize is not None
                culls = culls | (self.max_2Dsize > self.ctrl_cfg.cull_screen_size).squeeze()
        self._means = Parameter(self._means[~culls].detach())
        self._scales = Parameter(self._scales[~culls].detach())
        self._quats = Parameter(self._quats[~culls].detach())
        # self.colors_all = Parameter(self.colors_all[~culls].detach())
        self._features_dc = Parameter(self._features_dc[~culls].detach())
        self._features_rest = Parameter(self._features_rest[~culls].detach())
        self._opacities = Parameter(self._opacities[~culls].detach())

        print(f"     Cull: {n_bef - self.num_points}")
        return culls

    def split_gaussians(self, split_mask: torch.Tensor, samps: int) -> Tuple:
        """
        This function splits gaussians that are too large
        """

        n_splits = split_mask.sum().item()
        print(f"    Split: {n_splits}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            self.get_scaling[split_mask].repeat(samps, 1) * centered_samples
            # torch.exp(self._scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quat_act(self._quats[split_mask])  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self._means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        # new_colors_all = self.colors_all[split_mask].repeat(samps, 1, 1)
        new_feature_dc = self._features_dc[split_mask].repeat(samps, 1)
        new_feature_rest = self._features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self._opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        if self._scale_xy_limits[0] > 0:
            smaller = encode_road_scales(self.get_scaling[split_mask] / size_fac, self._scale_xy_limits)
        else:
            smaller = torch.log(torch.exp(self._scales[split_mask]) / size_fac)
        new_scales = smaller.repeat(samps, 1)
        self._scales[split_mask] = smaller
        # step 5, sample new quats
        new_quats = self._quats[split_mask].repeat(samps, 1)
        return new_means, new_feature_dc, new_feature_rest, new_opacities, new_scales, new_quats

    def dup_gaussians(self, dup_mask: torch.Tensor) -> Tuple:
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        print(f"      Dup: {n_dups}")
        dup_means = self._means[dup_mask]
        # dup_colors = self.colors_all[dup_mask]
        dup_feature_dc = self._features_dc[dup_mask]
        dup_feature_rest = self._features_rest[dup_mask]
        dup_opacities = self._opacities[dup_mask]
        dup_scales = self._scales[dup_mask]
        dup_quats = self._quats[dup_mask]
        return dup_means, dup_feature_dc, dup_feature_rest, dup_opacities, dup_scales, dup_quats

    def get_gaussians(self, cam: dataclass_camera) -> Dict:
        filter_mask = torch.ones_like(self._means[:, 0], dtype=torch.bool)
        self.filter_mask = filter_mask
        
        # get colors of gaussians
        colors = torch.cat((self._features_dc[:, None, :], self._features_rest), dim=1)
        if self.sh_degree > 0:
            viewdirs = self._means.detach() - cam.camtoworlds.data[..., :3, 3]  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = min(self.step // self.ctrl_cfg.sh_degree_interval, self.sh_degree)
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, 0.0, 1.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])
            
        activated_opacities = self.get_opacity
        activated_scales = self.get_scaling
        activated_rotations = self.get_quats
        actovated_colors = rgbs
        
        # collect gaussians information
        gs_dict = dict(
            _means=self._means[filter_mask],
            _opacities=activated_opacities[filter_mask],
            _rgbs=actovated_colors[filter_mask],
            _scales=activated_scales[filter_mask],
            _quats=activated_rotations[filter_mask],
        )
        
        # check nan and inf in gs_dict
        for k, v in gs_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in gaussian {k} at step {self.step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in gaussian {k} at step {self.step}")
                
        return gs_dict
    
    def compute_reg_loss(self):
        loss_dict = {}
        sharp_shape_reg_cfg = self.reg_cfg.get("sharp_shape_reg", None)
        if sharp_shape_reg_cfg is not None:
            w = sharp_shape_reg_cfg.w
            max_gauss_ratio = sharp_shape_reg_cfg.max_gauss_ratio
            step_interval = sharp_shape_reg_cfg.step_interval
            if self.step % step_interval == 0:
                # scale regularization
                scale_exp = self.get_scaling
                scale_reg = torch.maximum(scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(max_gauss_ratio)) - max_gauss_ratio
                scale_reg = scale_reg.mean() * w
                loss_dict["sharp_shape_reg"] = scale_reg

        flatten_reg = self.reg_cfg.get("flatten", None)
        if flatten_reg is not None:
            sclaings = self.get_scaling
            min_scale, _ = torch.min(sclaings, dim=1)
            min_scale = torch.clamp(min_scale, 0, 30)
            flatten_loss = torch.abs(min_scale).mean()
            loss_dict["flatten"] = flatten_loss * flatten_reg.w
        
        sparse_reg = self.reg_cfg.get("sparse_reg", None)
        if sparse_reg:
            if (self.cur_radii > 0).sum():
                opacity = (self.get_opacity if self.reg_cfg.get("alpha_min", None) is not None
                           else torch.sigmoid(self._opacities))
                opacity = opacity.clamp(1e-6, 1-1e-6)
                log_opacity = opacity * torch.log(opacity)
                log_one_minus_opacity = (1-opacity) * torch.log(1 - opacity)
                sparse_loss = -1 * (log_opacity + log_one_minus_opacity)[self.cur_radii > 0].mean()
                loss_dict["sparse_reg"] = sparse_loss * sparse_reg.w

        # compute the max of scaling
        max_s_square_reg = self.reg_cfg.get("max_s_square_reg", None)
        if max_s_square_reg is not None and not self.ball_gaussians:
            loss_dict["max_s_square"] = torch.mean((self.get_scaling.max(dim=1).values) ** 2) * max_s_square_reg.w
        return loss_dict
    
    def load_state_dict(self, state_dict: Dict, **kwargs) -> str:
        state_dict = state_dict.copy()
        saved_bounds = state_dict.get("_scale_xy_bounds", (-1., -1.))
        if tuple(float(v) for v in saved_bounds) != self._scale_xy_limits:
            if self.ball_gaussians or self.gaussian_2d:
                raise ValueError("Cannot convert smooth XY scales to ball/2D RoadNodes")
            physical = decode_road_scales(state_dict["_scales"], saved_bounds)
            state_dict["_scales"] = encode_road_scales(physical, self._scale_xy_limits)
            logger.info("Converted RoadNodes checkpoint XY scale parameterization")
        state_dict["_scale_xy_bounds"] = self._scale_xy_bounds.clone()
        saved_alpha_min = float(state_dict.get("_opacity_alpha_min", -1.0))
        current_alpha_min = float(self._opacity_alpha_min)
        if saved_alpha_min != current_alpha_min:
            raw = state_dict["_opacities"]
            opacity = (bounded_sigmoid(raw, saved_alpha_min)
                       if saved_alpha_min >= 0.0 else torch.sigmoid(raw))
            if saved_alpha_min < 0.0:
                # Legacy checkpoints used the configured hard opacity floor.
                lowest = self.reg_cfg.get("legacy_opacity_floor", self.reg_cfg.get("lowest_opacity", None))
                if lowest is not None:
                    opacity = opacity.clamp(min=float(lowest))
            state_dict["_opacities"] = self._opacity_to_raw(opacity)
            logger.info("Converted RoadNodes checkpoint opacity parameterization")
        state_dict["_opacity_alpha_min"] = self._opacity_alpha_min.clone()
        N = state_dict["_means"].shape[0]
        state_dict.setdefault("_hf_detail_mask", torch.zeros(N, dtype=torch.bool, device=self.device))
        self._hf_detail_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._means = Parameter(torch.zeros((N,) + self._means.shape[1:], device=self.device))
        self._scales = Parameter(torch.zeros((N,) + self._scales.shape[1:], device=self.device))
        self._quats = Parameter(torch.zeros((N,) + self._quats.shape[1:], device=self.device))
        self._features_dc = Parameter(torch.zeros((N,) + self._features_dc.shape[1:], device=self.device))
        self._features_rest = Parameter(torch.zeros((N,) + self._features_rest.shape[1:], device=self.device))
        self._opacities = Parameter(torch.zeros((N,) + self._opacities.shape[1:], device=self.device))
        msg = super().load_state_dict(state_dict, **kwargs)
        self._register_hf_gradient_hooks()
        if self.init_cfg.get("high_frequency", {}).get("enabled", False) and not self._hf_detail_mask.any():
            logger.warning("Checkpoint has no HF detail points. Resuming preserves its centers; "
                           "edge-guided initialization only runs when training from scratch.")
        return msg
    
    def export_gaussians_to_ply(self, alpha_thresh: float) -> Dict:
        means = self._means
        direct_color = self.colors
        
        activated_opacities = self.get_opacity
        mask = activated_opacities.squeeze() > alpha_thresh
        return {
            "positions": means[mask],
            "colors": direct_color[mask],
        }
