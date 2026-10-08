from typing import Dict, Optional
import torch
import logging
from omegaconf import OmegaConf

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere

logger = logging.getLogger()

import os
from utils.misc import export_points_to_ply, import_str

DEBUG_PCD = False
if DEBUG_PCD:
    DEBUG_OUTPUT_DIR = "debug"
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RoadNodes" in self.model_config:
            self.gaussian_classes['RoadNodes'] = GSModelType.RoadNodes
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
        if "TrafficLightNodes" in self.model_config:
            self.gaussian_classes["TrafficLightNodes"] = GSModelType.TrafficLightNodes
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes:
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
            if class_name in self.gaussian_classes.keys():
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device,
                    particle_cfg=self.particle_cfg,
                    weather_type=self.weather_type,
                )
                
            if class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)

            self.models[class_name] = model
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def _configure_road_background_prior(self) -> None:
        if "Background" not in self.models or "RoadNodes" not in self.models:
            return
        background = self.models["Background"]
        if hasattr(background, "set_road_surface_prior"):
            background.set_road_surface_prior(self.models["RoadNodes"]._means)

    def load_state_dict(self, *args, **kwargs):
        message = super().load_state_dict(*args, **kwargs)
        self._configure_road_background_prior()
        return message

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict, trafficlightnode_pts_dict = {}, {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )

        if "TrafficLightNodes" in self.model_config:
            trafficlightnode_pts_dict = dataset.get_init_objects(
                cur_node_type='TrafficLightNodes',
                **self.model_config["TrafficLightNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict, **trafficlightnode_pts_dict}
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False

            # NOTE(gls): 路面点云依赖于背景点云，因此路面点云和背景点云获取都放在Background中
            # 背景点云获取：雷达随机采样+去除动态box+远近随机初始化+过滤地面节点+过滤地面以下节点
            # 地面点云获取：雷达随机采样+去除动态box+提取地面节点(雷达点云pmf & road mask)
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    print("without from lidar!")
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None

                if DEBUG_PCD:
                    export_points_to_ply(
                        sampled_pts,
                        sampled_color,
                        save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_samples_pts.ply"),
                    )

                processed_pts_wo_box = dataset.project_aggregated_lidar_pts_wo_box(
                    delete_out_of_view_points=True
                )
                if DEBUG_PCD:
                    export_points_to_ply(
                        processed_pts_wo_box["pts"],
                        processed_pts_wo_box["colors"],
                        save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_samples_wo_box_pts.ply"),
                    )        
                # 点云分割类型：LIDAR (PMF算法) / IMAGE (相机road_mask) / LIDAR_AND_IMAGE (两者结合, 效果最佳)
                SEG_TYPE = init_cfg.get("seg_road_type", "LIDAR")
                if SEG_TYPE == "LIDAR":
                    processed_init_wo_road_pts, processed_init_road_pts = dataset.filter_pts_in_road( 
                        seed_pts=processed_pts_wo_box["pts"],
                        seed_colors=processed_pts_wo_box["colors"],
                        TYPE = "LIDAR",
                        road_only= True
                    )
                elif SEG_TYPE == "IMAGE":
                    processed_init_wo_road_pts, processed_init_road_pts = dataset.filter_pts_in_road( 
                        seed_pts=processed_pts_wo_box["pts"],
                        seed_colors=processed_pts_wo_box["colors"],
                        TYPE = "IMAGE",
                        road_only = True
                    )
                elif SEG_TYPE == "LIDAR_AND_IMAGE":
                    processed_init_wo_road_pts, processed_init_road_pts = dataset.filter_pts_in_road(
                        seed_pts=processed_pts_wo_box["pts"],
                        seed_colors=processed_pts_wo_box["colors"],
                        TYPE = "LIDAR_AND_IMAGE",
                        road_only = True
                    )
                
                if DEBUG_PCD:
                    export_points_to_ply(
                        processed_init_wo_road_pts["pts"],
                        processed_init_wo_road_pts["colors"],
                        save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_samples_wo_box_wo_road_pts.ply"),
                    )
                    export_points_to_ply(
                        processed_init_road_pts["pts"],
                        processed_init_road_pts["colors"],
                        save_path=os.path.join(DEBUG_OUTPUT_DIR, "lidar_samples_wo_box_road_pts.ply"),
                    )

                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
    
                #NOTE(gls): 背景点云方式是先去掉物体box内的点云，再去掉路面点云及以下，剩下的点云作为背景初始化点云
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )   
                processed_env_init_pts, _ = dataset.filter_pts_in_road(
                    seed_pts=processed_init_pts["pts"],
                    seed_colors=processed_init_pts["colors"],
                    TYPE = SEG_TYPE,
                    road_only = False
                )

                if DEBUG_PCD:
                    export_points_to_ply(
                        processed_env_init_pts["pts"],
                        processed_env_init_pts["colors"],
                        save_path=os.path.join(DEBUG_OUTPUT_DIR, "env_init_pts.ply"),
                    )
                model.create_from_pcd(
                    init_means=processed_env_init_pts["pts"], init_colors=processed_env_init_pts["colors"]
                )

            if class_name == "RoadNodes":
                lane_samples = None
                if model.ctrl_cfg.get("lane_influence_radius", None) is not None:
                    from datasets.qcraft.bevrg_utils import load_lane_line_samples
                    lane_samples = load_lane_line_samples(
                        dataset.data_path, dataset.start_timestep,
                        dataset.end_timestep, self.device,
                    )
                empty = model.create_from_pcd(
                    init_means=processed_init_road_pts["pts"],
                    init_colors=processed_init_road_pts["colors"],
                    lane_samples=lane_samples,
                )
                if not empty:
                    logger.info(f"RoadNodes Initialized Finish!")

            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
                
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )

            if class_name == 'TrafficLightNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=trafficlightnode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")

        self._configure_road_background_prior()
        logger.info(f"Initialized gaussians from pcd")
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False,
        render_only: bool = False,
        movement: torch.Tensor = torch.eye(4),
        front_cam: bool = True,
        first_frame: bool = False,
        edit_cfg: OmegaConf = None,
        freeze_road: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models

        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
                
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )

        if not front_cam:
            movement = torch.eye(4) 
        # processed_cam.camtoworlds[2,3] = processed_cam.camtoworlds[2,3] #+ 5

        if edit_cfg != None:
            gs, freeze_mask = self.collect_gaussians(
                cam=processed_cam,
                image_ids=image_infos["img_idx"].flatten()[0],
                movement=movement,
                front_cam=front_cam,
                first_frame=first_frame,
                edit_cfg=edit_cfg,
            )
        else:
            # NOTE(gls): 封锁RoadNode的xyz梯度,freeze_mask为路面部分的高斯mask
            gs, freeze_mask = self.collect_gaussians( # NOTE(gls): fix some road gaussian xyz by road mask 注意冻结的路面是true 需要区分出哪些是路面的高斯
                cam=processed_cam,
                image_ids=image_infos["img_idx"].flatten()[0],
            )  

        # 纯光栅化轨迹渲染专用，不计算复杂梯度项
        if render_only:
            if self.training:
                raise ValueError(
                    "render_only=True is incompatible with training mode; "
                    "it skips gradient setup and makes parameter updates a no-op. "
                    "Use render_only only for export/video rendering."
                )
            outputs, render_fn = self.render_gaussians(
                gs=gs,
                cam=processed_cam,
                near_plane=self.render_cfg.near_plane,
                far_plane=self.render_cfg.far_plane,
                render_mode="RGB+ED",
                radius_clip=self.render_cfg.get('radius_clip', 0.)
            )
            
            # 渲染天空并与高斯结果融合
            sky_rgb = self.models['Sky'](image_infos)
            sky_rgb_blend = sky_rgb * (1.0 - outputs["opacity"])
            outputs["rgb"] = self.affine_transformation(
                outputs["rgb_gaussians"] + sky_rgb_blend, image_infos
            )
            return outputs, gs

        if self.training:
            # 强制设为 True,这是注册钩子的前提条件
            gs.means.requires_grad_(True)
            gs.quats.requires_grad_(True)
            gs.scales.requires_grad_(True)
            gs.opacities.requires_grad_(True)

            if freeze_road:
                def zero_grad_for_road(grad): # grad (N, 3)
                    # freeze_mask 是 True 的位置是路面，想把这些位置的梯度清零
                    # ~freeze_mask 是 True 的位置是非路面，这些位置的梯度保持不变, 作为掩码来保留非路面的梯度
                    inverted_mask = ~freeze_mask
                    mask_float = inverted_mask.float()[..., None] # mask_float (N, 1)
                    return grad * mask_float

                params_to_register = {
                    'means': gs.means,
                    'quats': gs.quats,
                    # 'scales': gs.scales,  # 尺度用另外一个钩子函数限制z
                    # 'opacities': gs.opacities  # 透明度放宽
                }

                for name, param in params_to_register.items():
                    # 检查是否已经注册过（通过在 gs 对象上打标记）
                    attr_name = f'road_freeze_hook_registered_{name}'
                    if not hasattr(gs, attr_name):
                        param.register_hook(zero_grad_for_road)
                        # 标记为已注册
                        setattr(gs, attr_name, True)

                def scale_grad_hook(grad):
                    # 只针对路面部分 (freeze_mask == True) 进行处理
                    new_grad = grad.clone()
                    # 将路面高斯的 z 轴梯度 (index 2) 设为 0, 同时保留路面高斯的 x, y 轴梯度
                    new_grad[freeze_mask, 2] = 0.0 
                    return new_grad

                # 在初始化或 collect_gaussians 后注册
                gs.scales.register_hook(scale_grad_hook)
                attr_name_scales = f'road_freeze_hook_registered_scales'
                setattr(gs, attr_name_scales, True)

        # render gaussians
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )

        # NOTE(gls): 对output['rgb']增加一个road mask，然后把路面独立出来。
        # NOTE(ga): render_only / novel-view 路径可不加载 road_masks，此时跳过 road_rgb。
        if "road_masks" in image_infos:
            road_mask = image_infos["road_masks"]
            outputs['road_rgb'] = outputs['rgb'] * road_mask[..., None]
        outputs["freeze_mask"] = freeze_mask # 带回路面标注

        road_coverage_cfg = self.losses_dict.get("road_coverage", None)
        if (
            self.training
            and road_coverage_cfg is not None
            and road_coverage_cfg.get("w", 0.0) > 0
            and self.step > road_coverage_cfg.get("start_after", 0)
            and image_infos.get("road_masks") is not None
            and "RoadNodes" in self.gaussian_classes
        ):
            road_gaussian_mask = (
                self.pts_labels == self.gaussian_classes["RoadNodes"]
            )
            _, _, road_opacity = render_fn(road_gaussian_mask)
            outputs["RoadNodes_opacity"] = road_opacity

        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                    sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                    outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                    outputs[class_name+"_opacity"] = sep_opacity
                    outputs[class_name+"_depth"] = sep_depth

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                sep_rgb, sep_depth, sep_opacity = render_fn(gaussian_mask)
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
        return outputs, gs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
        gs
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos, gs)
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict
