from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import os
import time
import logging
from omegaconf import ListConfig

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import kornia
from enum import IntEnum
import viser
import nerfview
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.gaussians.basics import *

logger = logging.getLogger()


class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3
    TrafficLightNodes = 4
    RoadNodes = 5


def lr_scheduler_fn(cfg: OmegaConf, lr_init: float):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.0

        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func


def _normalize_list(value, num_items, item_name="item"):
    """将单个值或列表统一为指定长度的列表。"""
    if value is None:
        return [None] * num_items

    # 转换 OmegaConf 类型为原生 Python 类型
    if isinstance(value, ListConfig):
        value = list(value)

    if isinstance(value, (str, int, float)):
        return [value] * num_items

    if isinstance(value, list):
        if len(value) == 1 and num_items > 1:
            return value * num_items
        if len(value) != num_items:
            raise ValueError(
                f"{item_name} 数量 ({len(value)}) 与目标数量 ({num_items}) 不匹配"
            )
        return value

    raise ValueError(f"不支持的 {item_name} 类型: {type(value)}")


def _parse_waypoints(raw_waypoints, num_items, item_name="item"):
    """统一解析 waypoints，支持 (K,3)、(1,K,3)、(N,K,3) 等多种格式。"""
    if raw_waypoints is None:
        return [None] * num_items

    # 处理 OmegaConf 的 ListConfig
    if isinstance(raw_waypoints, ListConfig):
        raw_waypoints = list(raw_waypoints)

    wp = np.array(raw_waypoints)

    # 去掉多余的单维度: (1, K, 3) -> (K, 3)
    while wp.ndim > 2 and wp.shape[0] == 1:
        wp = wp.squeeze(0)

    if wp.ndim == 2:
        return [wp.tolist()] * num_items
    elif wp.ndim == 3:
        if wp.shape[0] != num_items:
            raise ValueError(
                f"waypoints 数量 ({wp.shape[0]}) 与 {item_name} 数量 ({num_items}) 不匹配"
            )
        return [w.tolist() for w in wp]
    else:
        raise ValueError(f"waypoints 格式错误，期望 (K,3) 或 (N,K,3)，实际 {wp.shape}")


class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
        particle_cfg: OmegaConf = None,
        weather_type: str = None,
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.step = 0
        self.device = device
        # init weather particle
        self.particle_cfg = particle_cfg
        self.weather_type = weather_type

        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images

        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)

        # init models
        self.models = {}
        self.misc_classes_keys = ["Sky", "Affine", "CamPose", "CamPosePerturb"]
        self.gaussian_classes = {}
        self._init_models()
        self.pts_labels = None  # will be overwritten in forward
        self.render_dynamic_mask = False

        # init losses fn
        self._init_losses()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
            self.device
        )
        self.step = 0

        # background color
        self.back_color = torch.zeros(3).to(self.device)

        # for evaluation
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices  # will be override

        # a simple viewer for background visualization
        self.viewer = None

        # keep gaussian
        self.gs = None

    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices

    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()

    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (
                    self.res_schedule.downscale_times
                    - self.step // self.res_schedule.double_steps
                ),
                0,
            )
        else:
            return 1

    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get("optim", None)
        class_ctrl_cfg = model_cfg.get("ctrl", None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg["optim"] = new_optim_cfg
        model_cfg["ctrl"] = new_ctrl_cfg

        return model_cfg

    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")

    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")

    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())

        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]

            raw_optim_cfg = class_optim_cfg.get(component_name, None)
            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create(
                {
                    "lr": raw_optim_cfg.get("lr", 0.0005),
                    "eps": raw_optim_cfg.get("eps", 1.0e-15),
                    "weight_decay": raw_optim_cfg.get("weight_decay", 0),
                }
            )
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert (
                optim_cfg is not None
            ), f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append(
                {
                    "params": params,
                    "name": params_name,
                    "lr": optim_cfg.lr,
                    "eps": optim_cfg.eps,
                    "weight_decay": optim_cfg.weight_decay,
                }
            )

            if raw_optim_cfg.get("lr_final", None) is not None:
                sched_cfg = OmegaConf.create(
                    {
                        "opt_after": raw_optim_cfg.get("opt_after", 0),
                        "warmup_steps": raw_optim_cfg.get("warmup_steps", 0),
                        "max_steps": raw_optim_cfg.get("max_steps", self.num_iters),
                        "lr_pre_warmup": raw_optim_cfg.get("lr_pre_warmup", 1.0e-8),
                        "lr_final": raw_optim_cfg.get("lr_final", None),
                        "ramp": raw_optim_cfg.get("ramp", "cosine"),
                    }
                )
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = (
                    sched_cfg.lr_final * lr_scale_factor
                    if sched_cfg.lr_final is not None
                    else None
                )
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(
            enabled=self.optim_general.get("use_grad_scaler", False)
        )

    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy

                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(
                    pred, gt, reduction="mean"
                )
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy

                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(
                    pred, gt, limit=0.1, reduction="mean"
                )
        self.sky_opacity_loss_fn = sky_opacity_loss_fn

        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss

            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )
        self.depth_loss_fn = depth_loss_fn

        if "RoadNodes" in self.models:
            road_loss_fn = None
            road_loss_cfg = self.losses_dict.get("road", None)
            if road_loss_cfg is not None:
                from models.losses import RoadLoss

                road_loss_fn = RoadLoss()  # TODO(gls): args
                self.road_loss_fn = road_loss_fn

    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def optimizer_step(self) -> None:
        # for params_name, optimizer in self.optimizers.items():
        #     class_name = params_name.split("#")[0]
        #     component_name = params_name.split("#")[1]
        #     max_norm = self.model_config[class_name]["optim"][component_name].get("max_norm", None)
        #     if max_norm is not None:
        #         self.grad_scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(self.param_groups[params_name], max_norm)
        #     if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
        #         self.grad_scaler.step(optimizer)
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()

    def postprocess_per_train_step(self, step: int) -> None:
        radii = self.info["radii"]
        if self.render_cfg.absgrad:
            grads = self.info["means2d"].absgrad.clone()
        else:
            grads = self.info["means2d"].grad.clone()
        grads[..., 0] *= self.info["width"] / 2.0 * self.render_cfg.batch_size
        grads[..., 1] *= self.info["height"] / 2.0 * self.render_cfg.batch_size

        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]

            self.models[class_name].postprocess_per_train_step(
                step=step,
                optimizer=self.optimizer,
                radii=radii[0, gaussian_mask],
                xys_grad=grads[0, gaussian_mask],
                last_size=max(self.info["width"], self.info["height"]),
            )

        # viewer
        if self.viewer is not None:
            num_train_rays_per_step = (
                self.render_cfg.batch_size * self.info["width"] * self.info["height"]
            )
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            # Update the viewer state.
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            # Update the scene.
            self.viewer.update(step, num_train_rays_per_step)

    def update_visibility_filter(self) -> None:
        for class_name in self.gaussian_classes.keys():
            gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
            cur_radii = self.info["radii"][0, gaussian_mask]
            if hasattr(self.models[class_name], "filter_mask"):
                n_filtered = int(self.models[class_name].filter_mask.sum().item())
                cur_radii = reduce_per_gaussian(cur_radii, n_filtered, reduce="amax")
            self.models[class_name].cur_radii = cur_radii

    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False,
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]

        # 在原轨迹下使用矫正后的相机位姿
        if not novel_view:
            if "CamPosePerturb" in self.models.keys():
                camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

            if "CamPose" in self.models.keys():
                camtoworlds = self.models["CamPose"](camtoworlds, image_ids)

        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"],
        )

        return camera_dict

    def color_legend(self, edit_cfg, image_output_pth=None):
        legend_cfg = getattr(edit_cfg, "legend", None)
        if legend_cfg is None:
            return

        if getattr(legend_cfg, "rigid", False):
            self.models["RigidNodes"].color_legned_gaussians(
                image_output_pth=os.path.join(image_output_pth, "legend_rigid.png")
            )

        if getattr(legend_cfg, "smpl", False):
            self.models["SMPLNodes"].color_legned_gaussians(
                image_output_pth=os.path.join(image_output_pth, "legend_smpl.png")
            )

    def edit_gaussians(
        self,
        edit_cfg: OmegaConf = None,
        self_trajectory: Tensor = None,
        dataset = None,
    ):
        instance_bboxes_all = []

        for class_name in edit_cfg.Nodes:
            node_cfg = edit_cfg.Nodes[class_name]
            if node_cfg is None:
                print(f"[info] 跳过 {class_name}，配置为 None")
                continue

            # ===== 分段偏移轨迹编辑（smoothstep + 旋转补偿）=====
            if "trajectory_edit" in node_cfg.keys():
                cfg = node_cfg.trajectory_edit
                instance_ids = _normalize_list(cfg.instance_id, 1, "instance_id")
                offset_windows_list = [list(w) for w in cfg.offset_windows]

                for instance_id, offset_windows in zip(instance_ids, offset_windows_list):
                    self.models[class_name].edit_trajectory_interpolate(
                        instance_id=instance_id,
                        offset_windows=offset_windows,
                    )

            # ===== 删除实例 =====
            if "remove" in node_cfg.keys():
                self.models[class_name].remove_instances(
                    remove_id_list=node_cfg.remove.instance_id
                )

            # ===== 替换实例 =====
            if "replace" in node_cfg.keys():
                rep_cfg = node_cfg.replace
                assert len(rep_cfg.target_id) == len(rep_cfg.replace_obj)
                if class_name == "SMPLNodes":
                    motion_sources = _normalize_list(
                        getattr(rep_cfg, "replace_motion", None),
                        len(rep_cfg.target_id),
                        "replace_motion",
                    )
                    for target_id, replace_obj, motion_source in zip(
                        rep_cfg.target_id, rep_cfg.replace_obj, motion_sources
                    ):
                        if isinstance(replace_obj, str):
                            self.models[class_name].replace_instance_with_ply(
                                target_id=target_id,
                                ply_path=replace_obj,
                                motion_source=motion_source,
                            )
                        elif isinstance(replace_obj, int):
                            self.models[class_name].replace_instances(
                                replace_dict={target_id: replace_obj},
                                motion_source_dict={target_id: motion_source},
                            )
                        else:
                            raise ValueError("replace_obj must be an int or str")
                else:
                    for target_id, replace_obj in zip(
                        rep_cfg.target_id, rep_cfg.replace_obj
                    ):
                        if isinstance(replace_obj, str):
                            self.models[class_name].replace_instance_with_ply(
                                target_id=target_id, ply_path=replace_obj
                            )
                        elif isinstance(replace_obj, int):
                            self.models[class_name].replace_instances(
                                replace_dict={target_id: replace_obj}
                            )
                        else:
                            raise ValueError("replace_obj must be an int or str")

            # ===== 添加实例 =====
            if "add" in node_cfg.keys():
                add_cfg = node_cfg.add
                _ds = dataset if getattr(add_cfg, 'color_correct', False) else None
                _radius = getattr(add_cfg, 'brightness_radius', 3.0)
                if class_name == "Background":
                    for add_obj, offset, time_window, rotation in zip(
                        edit_cfg.Nodes[class_name].add.add_obj,
                        edit_cfg.Nodes[class_name].add.offset,
                        edit_cfg.Nodes[class_name].add.time_window,
                        edit_cfg.Nodes[class_name].add.rotation,
                    ):
                        instance_bboxes = self.models[class_name].add_instance_with_ply(
                            ply_path=add_obj,
                            offset=offset,
                            self_trajectory=self_trajectory,
                            time_window=time_window,
                            rotation=rotation,
                            dataset=_ds,
                            models=self.models,
                            brightness_radius=_radius,
                        )
                        instance_bboxes_all += instance_bboxes
                else:
                    ref_ids = add_cfg.ref_id
                    add_objs = add_cfg.add_obj

                    if class_name == "SMPLNodes":
                        offsets = add_cfg.offset
                        assert (
                            len(ref_ids) == len(add_objs) == len(offsets)
                        ), f"Length mismatch: ref_id({len(ref_ids)}), add_obj({len(add_objs)}), offset({len(offsets)})"

                        if "add_motion" not in add_cfg.keys():
                            raise ValueError(
                                "SMPLNodes.add requires add_motion in edit config"
                            )
                        add_motions = _normalize_list(
                            add_cfg.add_motion, len(ref_ids), "add_motion"
                        )
                        appear_frames_list = _normalize_list(
                            getattr(add_cfg, "appear_frames", None),
                            len(ref_ids),
                            "appear_frames",
                        )
                        if class_name not in self.gaussian_classes:
                            self.gaussian_classes[class_name] = GSModelType.SMPLNodes

                        for ref_id, add_obj, offset, add_motion, appear_frames in zip(
                            ref_ids, add_objs, offsets, add_motions, appear_frames_list
                        ):
                            if not isinstance(add_obj, str):
                                raise ValueError("SMPLNodes.add.add_obj must be a str")
                            self.models[class_name].add_instance_with_ply(
                                target_id=ref_id,
                                ply_path=add_obj,
                                offset=offset,
                                motion_path=add_motion,
                                appear_frames=appear_frames,
                                dataset=_ds,
                            )
                    else:
                        # RigidNodes：统一用 waypoints 替代 offset
                        num = len(ref_ids)
                        waypoints_list = _parse_waypoints(
                            getattr(add_cfg, "waypoints", None), num, "ref_id"
                        )
                        methods = _normalize_list(
                            getattr(add_cfg, "method", "linear"), num, "method"
                        )
                        if hasattr(add_cfg, "time_window"):
                            time_windows = _normalize_list(
                                add_cfg.time_window,
                                num,
                                "time_window",
                            )
                        else:
                            time_windows = [None] * num
                        assert (
                            len(ref_ids)
                            == len(add_objs)
                            == len(waypoints_list)
                            == len(methods)
                            == len(time_windows)
                        ), (
                            f"Length mismatch: "
                            f"ref_id({len(ref_ids)}), "
                            f"add_obj({len(add_objs)}), "
                            f"waypoints({len(waypoints_list)}), "
                            f"method({len(methods)}), "
                            f"time_window({len(time_windows)})"
                        )
                        for ref_id, add_obj, wp, method, tw in zip(
                            ref_ids,
                            add_objs,
                            waypoints_list,
                            methods,
                            time_windows,
                        ):
                            if isinstance(add_obj, str):
                                self.models[class_name].add_instance_with_ply(
                                    target_id=ref_id,
                                    ply_path=add_obj,
                                    waypoints=wp,
                                    method=method,
                                    ego_traj=self_trajectory,
                                    time_window=tw,
                                    dataset=_ds,
                                    models=self.models,
                                    brightness_radius=_radius,
                                )
                            elif isinstance(add_obj, int):
                                self.models[class_name].add_instance(
                                    target_id=ref_id,
                                    add_id=add_obj,
                                    waypoints=wp,
                                    method=method,
                                    ego_traj=self_trajectory,
                                    time_window=tw,
                                )
                            else:
                                raise ValueError("add_obj must be an int or str")

        return instance_bboxes_all

    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor,  # leave it here for future use
        movement: torch.Tensor = torch.eye(4),
        front_cam: bool = True,
        first_frame: bool = False,
        edit_cfg: OmegaConf = None,
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
        }
        for class_name in self.gaussian_classes.keys():
            if class_name == "Background" and edit_cfg != None:
                gs = self.models[class_name].get_gaussians_with_particles(
                    cam, movement, front_cam, first_frame
                )
            else:
                gs = self.models[class_name].get_gaussians(cam)

            if gs is None:
                continue

            # collect gaussians
            gs["class_labels"] = torch.full(
                (gs["_means"].shape[0],),
                self.gaussian_classes[class_name],
                device=self.device,
            )
            for k, _ in gs.items():
                if k != "legend_img":
                    gs_dict[k].append(gs[k])

        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)

        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = self.pts_labels != 0

        # --- 新增部分：创建并返回 freeze_mask ---
        # 1. 找到 "RoadNodes" 对应的类别 ID
        road_class_id = self.gaussian_classes.get("RoadNodes")

        if road_class_id is not None:
            # 2. 创建布尔掩码，路面高斯的位置为 True
            freeze_mask = self.pts_labels == road_class_id
        else:
            # 如果没有 "RoadNodes" 这个类别，则不冻结任何东西
            freeze_mask = torch.zeros_like(self.pts_labels, dtype=torch.bool)
        # --- 新增部分结束 ---

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],  # if "means" in detach_keys, then the means will be detached
            extras=None,  # to save some extra information (TODO) more flexible way
        )

        return gaussians, freeze_mask

    def render_gaussians(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        def render_fn(opaticy_mask=None, return_info=False):
            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=(
                    gs.opacities.squeeze() * opaticy_mask
                    if opaticy_mask is not None
                    else gs.opacities.squeeze()
                ),
                colors=gs.rgbs,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode=(
                    "antialiased" if self.render_cfg.antialiased else "classic"
                ),
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert (
                self.render_cfg.batch_size == 1
            ), "batch size must be 1, will support batch size > 1 in the future"

            assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
            rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)

            if not return_info:
                return (
                    torch.clamp(rendered_rgb, max=1.0),
                    rendered_depth,
                    alphas[..., None],
                )
            else:
                return (
                    torch.clamp(rendered_rgb, max=1.0),
                    rendered_depth,
                    alphas[..., None],
                    info,
                )

        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)
        results = {"rgb_gaussians": rgb, "depth": depth, "opacity": opacity}

        if self.training:
            self.info["means2d"].retain_grad()

        return results, render_fn

    def affine_transformation(
        self, rgb_blended: torch.Tensor, image_infos: Dict[str, torch.Tensor]
    ):
        if "Affine" in self.models:
            affine_trs = self.models["Affine"](image_infos)
            rgb_transformed = (
                affine_trs[..., :3, :3] @ rgb_blended[..., None]
                + affine_trs[..., :3, 3:]
            )[..., 0]

            return rgb_transformed
        else:
            return rgb_blended

    def forward(
        self,
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, "in_test_set"):
                model.in_test_set = self.in_test_set

        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view,
        )
        gs, _ = self.collect_gaussians(
            cam=processed_cam, image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get("radius_clip", 0.0),
        )

        # render sky
        sky_model = self.models["Sky"]
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])

        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]),
            image_infos,
        )

        return outputs

    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())
        self.grad_scaler.scale(total_loss).backward()
        self.optimizer_step()

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()

        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
        gs,
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}

        if "egocar_masks" in image_infos:
            # in the case of egocar, we need to mask out the egocar region
            valid_loss_mask = ~image_infos["egocar_masks"]
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"], dtype=torch.bool)

        ### NOTE(gls)： 在rgb的loss计算上，路面和非路面区域区分计算(实验验证分开略有增益)
        road_mask = image_infos["road_masks"]
        non_road_mask = ~road_mask

        gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
        gt_rgb = gt_rgb * non_road_mask[..., None]

        predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]
        predicted_rgb = predicted_rgb * non_road_mask[..., None]

        gt_occupied_mask = (~image_infos["sky_masks"] & valid_loss_mask).float()
        pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask

        ### NOTE(gls): 非路面区域的rgb_loss
        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(
            gt_rgb.permute(2, 0, 1)[None, ...],
            predicted_rgb.permute(2, 0, 1)[None, ...],
        )
        loss_dict.update(
            {
                "rgb_loss": self.losses_dict.rgb.w * Ll1,
                "ssim_loss": self.losses_dict.ssim.w * simloss,
            }
        )

        # mask loss
        if self.sky_opacity_loss_fn is not None:
            sky_loss_opacity = (
                self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask)
                * self.losses_dict.mask.w
            )
            loss_dict.update({"sky_loss_opacity": sky_loss_opacity})

        # depth loss
        if self.depth_loss_fn is not None:
            gt_depth = image_infos["lidar_depth_map"]
            lidar_hit_mask = (gt_depth > 0) & valid_loss_mask

            if self.depth_loss_fn.mask_out_dynamic_regions:
                lidar_hit_mask = lidar_hit_mask & ~image_infos["dynamic_masks"]

            if lidar_hit_mask.sum() > 0:
                pred_depth = outputs["depth"]
                depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)

                lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
                if lidar_w_decay > 0:
                    decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
                else:
                    decay_weight = 1
                depth_loss = depth_loss * self.losses_dict.depth.w * decay_weight
                loss_dict.update({"depth_loss": depth_loss})
            else:
                logger.debug("No valid lidar points for depth loss")

        # NOTE(gls): add road loss
        if self.road_loss_fn is not None:
            road_mask = image_infos["road_masks"]
            gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
            gt_road_rgb = gt_rgb * road_mask[..., None]
            pred_road_rgb = outputs["road_rgb"] * valid_loss_mask[..., None]

            # NOTE(gls)：路面相关参数可以单独传入(暂时不用，后面设计复杂loss可能启用)
            # freeze_mask = outputs["freeze_mask"]
            # road_quats = gs.quats[freeze_mask]
            # road_scales = gs.scales[freeze_mask]
            Ll1 = self.road_loss_fn(pred_road_rgb, gt_road_rgb)
            simloss_road = 1 - self.ssim(
                gt_road_rgb.permute(2, 0, 1)[None, ...],
                pred_road_rgb.permute(2, 0, 1)[None, ...],
            )
            road_loss = Ll1 * self.losses_dict.road.w
            simloss_road = simloss_road * self.losses_dict.ssim.w
            loss_dict.update({"road_loss": road_loss})
            loss_dict.update({"ssim_road_loss": simloss_road})

        road_coverage_cfg = self.losses_dict.get("road_coverage", None)
        if (
            road_coverage_cfg is not None
            and "RoadNodes_opacity" in outputs
            and image_infos.get("road_masks") is not None
            and float(road_coverage_cfg.get("w", 0.0)) > 0
            and self.step > int(road_coverage_cfg.get("start_after", 0))
        ):
            from models.losses import erode_binary_mask, road_coverage_loss

            valid_road_mask = image_infos["road_masks"] & valid_loss_mask
            if image_infos.get("dynamic_masks") is not None:
                valid_road_mask &= ~image_infos["dynamic_masks"]
            valid_road_mask = erode_binary_mask(
                valid_road_mask,
                int(road_coverage_cfg.get("erosion_px", 3)),
            )

            if valid_road_mask.any():
                lane_mask = image_infos.get("lane_masks", None)
                if lane_mask is not None:
                    lane_mask = lane_mask & valid_road_mask
                coverage_loss = road_coverage_loss(
                    outputs["RoadNodes_opacity"],
                    valid_road_mask,
                    float(road_coverage_cfg.get("target_alpha", 0.95)),
                    lane_mask=lane_mask,
                    lane_target_alpha=float(
                        road_coverage_cfg.get("lane_target_alpha", 0.98)
                    ),
                )
                warmup_steps = int(road_coverage_cfg.get("warmup_steps", 0))
                if warmup_steps > 0:
                    start_after = int(road_coverage_cfg.get("start_after", 0))
                    warmup = min(
                        1.0, (self.step - start_after) / warmup_steps
                    )
                else:
                    warmup = 1.0
                loss_dict["road_coverage_loss"] = (
                    coverage_loss * float(road_coverage_cfg.get("w", 0.0)) * warmup
                )

        # 不透明度的熵正则化，其主要作用是鼓励不透明度预测值趋向于两极化（即非常接近0或非常接近1）。
        # ----- reg loss -----
        opacity_entropy_reg = self.losses_dict.get("opacity_entropy", None)
        if opacity_entropy_reg is not None:
            pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
            loss_dict.update(
                {
                    "opacity_entropy_loss": opacity_entropy_reg.w
                    * (-pred_opacity * torch.log(pred_opacity)).mean()
                }
            )

        # from pvg: https://github.com/fudan-zvg/PVG/blob/b4162a9135282e0f3c929054f16be1b3fbacd77a/train.py#L161
        inverse_depth_smoothness_reg = self.losses_dict.get(
            "inverse_depth_smoothness", None
        )
        if inverse_depth_smoothness_reg is not None:
            inverse_depth = 1 / (outputs["depth"] + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                image_infos["pixels"][None].permute(0, 3, 1, 2),
            )
            loss_dict.update(
                {
                    "inverse_depth_smoothness_loss": inverse_depth_smoothness_reg.w
                    * loss_inv_depth
                }
            )

        # affine reg loss
        affine_reg = self.losses_dict.get("affine", None)
        if affine_reg is not None and "Affine" in self.models:
            affine_trs = self.models["Affine"](
                {"img_idx": image_infos["img_idx"].flatten()[0]}
            )
            reg_mat = torch.eye(3, device=self.device)
            reg_shift = torch.zeros(3, device=self.device)
            loss_affine = (
                torch.abs(affine_trs[..., :3, :3] - reg_mat).mean()
                + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
            )
            loss_dict.update({"affine_loss": affine_reg.w * loss_affine})

        # dynamic region loss
        dynamic_region_weighted_losses = self.losses_dict.get("dynamic_region", None)
        if dynamic_region_weighted_losses is not None:
            weight_factor = dynamic_region_weighted_losses.get("w", 1.0)
            start_from = dynamic_region_weighted_losses.get("start_from", 0)
            if self.step == start_from:
                self.render_dynamic_mask = True
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask

                if dynamic_pred_mask.sum() > 0:
                    Ll1 = torch.abs(
                        gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]
                    ).mean()
                    loss_dict.update(
                        {
                            "vehicle_region_rgb_loss": weight_factor * Ll1,
                        }
                    )

        # compute gaussian reg loss
        for class_name in self.gaussian_classes.keys():
            class_reg_loss = self.models[class_name].compute_reg_loss()
            for k, v in class_reg_loss.items():
                loss_dict[f"{class_name}_{k}"] = v
        return loss_dict

    def compute_metrics(
        self, outputs: Dict[str, torch.Tensor], image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict

    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict

    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update(
            {
                "models": {k: v.state_dict() for k, v in self.models.items()},
                "step": self.step,
            }
        )
        if not only_model:
            state_dict.update(
                {
                    "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
                    # "lr_schedulers": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
                    # "grad_scaler": self.grad_scaler.state_dict(),
                }
            )
        return state_dict

    def load_state_dict(
        self, state_dict: dict, load_only_model: bool = True, strict: bool = True
    ):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)

        # load model
        model_state_dict = state_dict.pop("models")
        for class_name in list(self.models.keys()):
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                logger.warning(f"Cannot find {class_name} in the checkpoint, removing from models")
                del self.models[class_name]
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")

    def resume_from_checkpoint(
        self, ckpt_path: str, load_only_model: bool = True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, weights_only=False)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=True)

    def save_checkpoint(
        self, log_dir: str, save_only_model: bool = True, is_final: bool = False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")

    def init_viewer(self, port: int = 8080):
        # a simple viewer for background ONLY visualization
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        cam = dataclass_camera(camtoworlds=c2w, camtoworlds_gt=c2w, Ks=K, H=H, W=W)

        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }
        for class_name in ["Background"]:
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue

            for k, _ in gs.items():
                gs_dict[k].append(gs[k])

        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None,
        )

        render_colors, _, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        return render_colors[0].cpu().numpy()
