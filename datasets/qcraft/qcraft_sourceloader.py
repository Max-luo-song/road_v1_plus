from typing import Dict, Optional
import logging
import os
import json
import joblib
import numpy as np
from tqdm import trange, tqdm
from omegaconf import OmegaConf

import torch
from torch import Tensor

from pytorch3d.transforms import matrix_to_quaternion
from datasets.base.scene_dataset import ModelType
from datasets.base.lidar_source import SceneLidarSource
from datasets.base.pixel_source import ScenePixelSource, CameraData
from datasets.qcraft.qcraft_helpers import load_available_camera_ids, load_lidar2ego

logger = logging.getLogger()

# define each class's node type
OBJECT_CLASS_NODE_MAPPING = {
    "Vehicle": ModelType.RigidNodes,
    "Pedestrian": ModelType.SMPLNodes,
    "Cyclist": ModelType.RigidNodes,
    "TrafficLight": ModelType.TrafficLightNodes,
}
SMPLNODE_CLASSES = ["Pedestrian"]

# Qcraft Camera List:
# 0 : "front_wide_110",      广角前视 FOV110
# 1 : "front_wide_60",       广角前视 FOV60
# 2 : "front_tele_30",       长焦前视 FOV30
# 3 : "front_tele_15",       长焦前视 FOV15
# 4 : "front_wide_left_60",  广角左前 FOV60
# 5 : "front_left_99",       左前 FOV99
# 6 : "rear_left_99",        左后 FOV99
# 7 : "rear_left_30",        左后 FOV30
# 8 : "front_wide_right_60", 广角右前 FOV60
# 9 : "front_right_99",      右前 FOV99
# 10 : "rear_right_99",      右后 FOV99
# 11 : "rear_right_30",      右后 FOV30
# 12 : "rear_50",            后视 FOV50


class QcraftCameraData(CameraData):
    def __init__(self, **kwargs):
        self.cam_to_worlds = None
        self.intrinsics = None
        self.distortions = None
        super().__init__(**kwargs)
        if self.mix_novel_alias and self.mix_novel_alias:
            self.load_novel_pose()
        else:
            self.load_calibrations()

        self.to(self.device)

    def load_novel_pose(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, lidar-to-world matrices, etc.
        hard code here
        """
        this_intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.from_cam_id}.txt")
        )
        fx, fy, cx, cy = (
            this_intrinsic[0],
            this_intrinsic[1],
            this_intrinsic[2],
            this_intrinsic[3],
        )
        this_distortions = this_intrinsic[4:]

        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )

        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array(this_distortions)

        cam_to_worlds, lidar_to_worlds, intrinsics, distortions = [], [], [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first lidar pose as the origin of the world coordinate system.
        cam_first_pose = np.loadtxt(
            os.path.join(
                f"{self.data_path}/novel_views/{self.mix_novel_alias}",
                f"cam_pose/{self.start_timestep:06d}.txt",
            )
        )

        for t in range(self.start_timestep, self.end_timestep):
            cam2world = np.loadtxt(
                os.path.join(
                    f"{self.data_path}/novel_views/{self.mix_novel_alias}",
                    f"cam_pose/{t:06d}.txt",
                )
            )
            # compute lidar_to_world transformation
            cam2world = np.linalg.inv(cam_first_pose) @ cam2world
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

    def load_calibrations(self):
        """
        Load the camera intrinsics, extrinsics, timestamps, etc.
        Compute the camera-to-world matrices, lidar-to-world matrices, etc.
        """
        intrinsic = np.loadtxt(
            os.path.join(self.data_path, "intrinsics", f"{self.cam_id}.txt")
        )
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        distortions = intrinsic[4:]

        # scale intrinsics w.r.t. load size
        fx, fy = (
            fx * self.load_size[1] / self.original_size[1],
            fy * self.load_size[0] / self.original_size[0],
        )
        cx, cy = (
            cx * self.load_size[1] / self.original_size[1],
            cy * self.load_size[0] / self.original_size[0],
        )

        _intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        _distortions = np.array(distortions)

        # load camera extrinsics
        cam_to_main_lidar = np.loadtxt(
            os.path.join(self.data_path, "extrinsics", f"{self.cam_id}.txt")
        )
        # compute per-image poses and intrinsics
        cam_to_worlds = []
        intrinsics, distortions = [], []

        # we tranform the camera poses w.r.t. the first timestep to make the translation vector of
        # the first lidar pose as the origin of the world coordinate system.
        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:06d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            lidar_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "lidar_pose", f"{t:06d}.txt")
            )
            # compute lidar_to_world transformation
            lidar_to_world = (
                np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            )
            # transformation:
            #   (opencv_cam -> qcraft_cam -> qcraft_lidar) -> current_world
            cam2world = lidar_to_world @ cam_to_main_lidar
            cam_to_worlds.append(cam2world)
            intrinsics.append(_intrinsics)
            distortions.append(_distortions)

        self.intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0)).float()
        self.distortions = torch.from_numpy(np.stack(distortions, axis=0)).float()
        self.cam_to_worlds = torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()

        self.cam_to_ego = None
        if self.render_only:
            main_lidar_to_ego = load_lidar2ego(self.data_path)
            self.cam_to_ego = main_lidar_to_ego @ cam_to_main_lidar  # syc
    
    @classmethod
    def get_camera2worlds(
        cls, data_path: str, cam_id: str, start_timestep: int, end_timestep: int
    ) -> torch.Tensor:
        """
        Returns camera-to-world matrices for the specified camera and time range.

        Args:
            data_path (str): Path to the dataset.
            cam_id (str): Camera ID.
            start_timestep (int): Start timestep.
            end_timestep (int): End timestep.

        Returns:
            torch.Tensor: Camera-to-world matrices of shape (num_frames, 4, 4).
        """
        # Load camera extrinsics
        cam_to_lidar = np.loadtxt(
            os.path.join(data_path, "extrinsics", f"{cam_id}.txt")
        )

        # Load lidar poses and compute camera-to-world matrices
        cam_to_worlds = []
        lidar_to_world_start = np.loadtxt(
            os.path.join(data_path, "lidar_pose", f"{start_timestep:06d}.txt")
        )

        for t in range(start_timestep, end_timestep):
            lidar_to_world_current = np.loadtxt(
                os.path.join(data_path, "lidar_pose", f"{t:06d}.txt")
            )
            lidar_to_world = (
                np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            )
            cam2world = lidar_to_world @ cam_to_lidar
            cam_to_worlds.append(cam2world)

        return torch.from_numpy(np.stack(cam_to_worlds, axis=0)).float()


class QcraftPixelSource(ScenePixelSource):
    def __init__(
        self,
        dataset_name: str,
        pixel_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
        render_only: bool = False,
        weather_type: str = None,
    ):
        super().__init__(dataset_name, pixel_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.render_only = render_only  # syc
        self.weather_type = weather_type
        self.mix_novel_views, self.mix_novel_cams = False, []
        if pixel_data_config.get("mix_novel_views", False):
            self.mix_novel_views = pixel_data_config.mix_novel_views
            self.mix_novel_cams = pixel_data_config.mix_novel_cams

        if render_only:
            # 计算新轨迹，因此加载 ego 轨迹
            self.load_ego_poses()
            
        self.load_data()

        if not render_only and pixel_data_config.get("load_lane_mask", False):
            from datasets.qcraft.lane_supervision import attach_lane_masks
            attach_lane_masks(self.camera_data, pixel_data_config)

    def load_ego_poses(self):
        """统一加载场景全局的自车轨迹"""
        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:06d}.txt")
        )
        ego_to_worlds = []
        for t in range(self.start_timestep, self.end_timestep):
            ego_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "ego_pose", f"{t:06d}.txt")
            )
            ego_to_world = np.linalg.inv(lidar_to_world_start) @ ego_to_world_current
            ego_to_worlds.append(ego_to_world)
        self.ego_to_worlds = torch.from_numpy(np.stack(ego_to_worlds, axis=0)).float().to(self.device)
        logger.info(f"Loaded {len(ego_to_worlds)} ego poses for the scene.")

    def load_cameras(self):
        self._timesteps = torch.arange(
            self.start_timestep, self.end_timestep
        )  ### 生成等差数列，目的是获取序列时间，对每一个时间都加载相机
        self.register_normalized_timestamps()  ### 归一化时间戳

        total_length = (
            len(self.camera_list) + len(self.mix_novel_cams)
            if self.mix_novel_views
            else len(self.camera_list)
        )

        for idx, cam_id in enumerate(
            self.camera_list
        ):  ### camera_list就是config中相机列表[0,1,2,3,4]
            logger.info(f"Loading camera {cam_id}")
            camera = QcraftCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                load_sky_mask=self.data_cfg.load_sky_mask,
                load_road_mask=self.data_cfg.load_road_mask,
                downscale_when_loading=self.data_cfg.downscale_when_loading[idx],
                undistort=self.data_cfg.undistort,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                render_only=self.render_only,
                weather_type=self.weather_type,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = (
                torch.arange(len(camera), device=self.device) * total_length + idx
            )
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"Camera {camera.cam_name} loaded.")
            self.camera_data[cam_id] = camera

        if self.mix_novel_views and self.mix_novel_cams:
            for idx, cam_alias in enumerate(self.mix_novel_cams):
                # increase id from last normal cam_id
                this_cam_id = self.camera_list[-1] + idx + 1
                logger.info(f"Loading mixed novel camera {cam_alias}")
                same_ds_idx = int(cam_alias.split("_")[-1])
                camera = QcraftCameraData(
                    dataset_name=self.dataset_name,
                    data_path=self.data_path,
                    cam_id=this_cam_id,
                    start_timestep=self.start_timestep,
                    end_timestep=self.end_timestep,
                    load_dynamic_mask=self.data_cfg.load_dynamic_mask,
                    load_sky_mask=self.data_cfg.load_sky_mask,
                    downscale_when_loading=self.data_cfg.downscale_when_loading[0],
                    undistort=self.data_cfg.undistort,
                    buffer_downscale=self.buffer_downscale,
                    device=self.device,
                    render_only=self.render_only,
                    mix_novel_views=self.mix_novel_views,
                    mix_novel_alias=cam_alias,
                )
                camera.load_time(self.normalized_time)
                unique_img_idx = (
                    torch.arange(len(camera), device=self.device) * total_length
                    + len(self.camera_list)
                    + idx
                )
                camera.set_unique_ids(
                    unique_cam_idx=len(self.camera_list) + idx,
                    unique_img_idx=unique_img_idx,
                )
                logger.info(f"Mixed Novel Camera {camera.cam_name} loaded.")
                self.camera_data[this_cam_id] = camera

    # syc
    def load_specified_cameras(
        self, cam_ids, downscale_when_loading
    ) -> Dict[int, CameraData]:
        camera_data = {}
        total_length = len(self.camera_list)

        for idx, cam_id in enumerate(cam_ids):
            absolute_idx = self.camera_list.index(cam_id)

            camera = QcraftCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                load_road_mask=self.data_cfg.load_road_mask,
                downscale_when_loading=downscale_when_loading[idx],
                undistort=False,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                render_only=True,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = (
                torch.arange(len(camera), device=self.device) * total_length + absolute_idx
            )
            camera.set_unique_ids(unique_cam_idx=absolute_idx, unique_img_idx=unique_img_idx)
            logger.info(f"Specified camera {camera.cam_name} loaded.")

            camera_data[cam_id] = camera

        return camera_data

    # bty
    def load_novel_cameras(
        self, cam_ids, downscale_when_loading
    ) -> Dict[int, CameraData]:
        camera_data = {}
        for idx, cam_id in enumerate(cam_ids):
            camera = QcraftCameraData(
                dataset_name=self.dataset_name,
                data_path=self.data_path,
                cam_id=cam_id,
                start_timestep=self.start_timestep,
                end_timestep=self.end_timestep,
                downscale_when_loading=downscale_when_loading[idx],
                undistort=False,
                buffer_downscale=self.buffer_downscale,
                device=self.device,
                render_only=True,
            )
            camera.load_time(self.normalized_time)
            unique_img_idx = (
                torch.arange(len(camera), device=self.device) * len(cam_ids) + idx
            )
            camera.set_unique_ids(unique_cam_idx=idx, unique_img_idx=unique_img_idx)
            logger.info(f"Specified camera {camera.cam_name} loaded.")

            camera_data[cam_id] = camera

        return camera_data

    def load_objects(self):
        """
        get ground truth bounding boxes of the dynamic objects

        instances_info = {
            "0": # simplified instance id
                {
                    "id": str,
                    "class_name": str,
                    "frame_annotations": {
                        "frame_idx": List,
                        "obj_to_world": List,
                        "box_size": List,
                },
            ...
        }
        frame_instances = {
            "0": # frame idx
                List[int] # list of simplified instance ids
            ...
        }
        """

        instances_info_path = os.path.join(
            self.data_path, "instances", "instances_info.json"
        )
        frame_instances_path = os.path.join(
            self.data_path, "instances", "frame_instances.json"
        )
        with open(instances_info_path, "r") as f:
            instances_info = json.load(f)
        with open(frame_instances_path, "r") as f:
            frame_instances = json.load(f)
        # get pose of each instance at each frame
        # shape (num_frames, num_instances, 4, 4)
        num_instances = len(instances_info)
        num_full_frames = len(frame_instances)
        instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
        instances_size = np.zeros((num_full_frames, num_instances, 3))
        instances_true_id = np.arange(num_instances)
        instances_model_types = np.ones(num_instances) * -1

        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:06d}.txt")
        )
        ### k最大有500+, 但instances_info只有119
        for index, (k, v) in enumerate(instances_info.items()):
            instances_model_types[int(index)] = OBJECT_CLASS_NODE_MAPPING[
                v["class_name"]
            ]
            for frame_idx, obj_to_world, box_size in zip(
                v["frame_annotations"]["frame_idx"],
                v["frame_annotations"]["obj_to_world"],
                v["frame_annotations"]["box_size"],
            ):
                # the first lidar pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(lidar_to_world_start) @ obj_to_world
                instances_pose[frame_idx, int(index)] = np.array(obj_to_world)
                instances_size[frame_idx, int(index)] = np.array(box_size)

        per_frame_instance_mask = np.zeros(
            (num_full_frames, num_instances)
        )  ### 实例个数

        ## TODO：把valid_instances里面的编号换成索引
        for frame_idx, valid_instances in frame_instances.items():
            temp_result = []
            for i in range(len(valid_instances)):
                for index, (k, v) in enumerate(instances_info.items()):
                    if k == str(valid_instances[i]):
                        temp_result.append(index)
                        break
            valid_instances = temp_result
            # valid_instances = change_index(valid_instances, instances_info)
            per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        # # waymo origin
        # for index, (k, v) in enumerate(instances_info.items()):
        #     instances_model_types[int(k)] = OBJECT_CLASS_NODE_MAPPING[v["class_name"]]
        #     for frame_idx, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"], v["frame_annotations"]["obj_to_world"], v["frame_annotations"]["box_size"]):
        #         # the first lidar pose as the origin of the world coordinate system.
        #         obj_to_world = np.array(obj_to_world).reshape(4, 4)
        #         obj_to_world = np.linalg.inv(lidar_to_world_start) @ obj_to_world
        #         instances_pose[frame_idx, int(k)] = np.array(obj_to_world)
        #         instances_size[frame_idx, int(k)] = np.array(box_size)

        # # get frame valid instances
        # # shape (num_frames, num_instances)
        # per_frame_instance_mask = np.zeros((num_full_frames, num_instances)) ### 实例个数
        # for frame_idx, valid_instances in frame_instances.items():
        #     per_frame_instance_mask[int(frame_idx), valid_instances] = 1

        # select the frames that are in the range of start_timestep and end_timestep
        instances_pose = torch.from_numpy(
            instances_pose[self.start_timestep : self.end_timestep]
        ).float()
        instances_size = torch.from_numpy(
            instances_size[self.start_timestep : self.end_timestep]
        ).float()
        instances_true_id = torch.from_numpy(instances_true_id).long()
        instances_model_types = torch.from_numpy(instances_model_types).long()
        per_frame_instance_mask = torch.from_numpy(
            per_frame_instance_mask[self.start_timestep : self.end_timestep]
        ).bool()

        # filter out the instances that are not visible in selected frames
        ins_frame_cnt = per_frame_instance_mask.sum(dim=0)
        instances_pose = instances_pose[:, ins_frame_cnt > 0]
        instances_size = instances_size[:, ins_frame_cnt > 0]
        instances_true_id = instances_true_id[ins_frame_cnt > 0]
        instances_model_types = instances_model_types[ins_frame_cnt > 0]
        per_frame_instance_mask = per_frame_instance_mask[:, ins_frame_cnt > 0]

        # assign to the class
        # (num_frames, num_instances, 4, 4)
        self.instances_pose = instances_pose
        # (num_instances, 3)
        self.instances_size = instances_size.sum(0) / per_frame_instance_mask.sum(
            0
        ).unsqueeze(-1)
        # (num_frames, num_instances)
        self.per_frame_instance_mask = per_frame_instance_mask
        # (num_instances)
        self.instances_true_id = instances_true_id
        # (num_instances)
        self.instances_model_types = instances_model_types

        if self.data_cfg.load_smpl:
            cam_ids = load_available_camera_ids(self.data_path)

            # Collect camera-to-world matrices for all available cameras
            cam_to_worlds = {}
            for cam_id in cam_ids:
                cam_to_worlds[cam_id] = QcraftCameraData.get_camera2worlds(
                    self.data_path, str(cam_id), self.start_timestep, self.end_timestep
                )

            # load SMPL parameters
            smpl_dict = joblib.load(
                os.path.join(self.data_path, "humanpose", "smpl.pkl")
            )
            frame_num = self.end_timestep - self.start_timestep

            smpl_human_all = {}
            for fi in tqdm(
                range(self.start_timestep, self.end_timestep), desc="Loading SMPL"
            ):
                for instance_id, ins_smpl in smpl_dict.items():
                    if instance_id not in smpl_human_all:
                        smpl_human_all[instance_id] = {
                            "smpl_quats": torch.zeros(
                                (frame_num, 24, 4), dtype=torch.float32
                            ),
                            "smpl_trans": torch.zeros(
                                (frame_num, 3), dtype=torch.float32
                            ),
                            "smpl_betas": torch.zeros(
                                (frame_num, 10), dtype=torch.float32
                            ),
                            "frame_valid": torch.zeros((frame_num), dtype=torch.bool),
                        }
                        smpl_human_all[instance_id]["smpl_quats"][:, :, 0] = 1.0
                    if ins_smpl["valid_mask"][fi]:
                        # 检查 frame_idx 中是否存在 fi，避免 SMPL 与实例标注不匹配
                        frame_idx_list = instances_info[str(instance_id)]["frame_annotations"]["frame_idx"]
                        if fi not in frame_idx_list:
                            continue
                        ii = frame_idx_list.index(fi)

                        betas = ins_smpl["smpl"]["betas"][fi]
                        smpl_human_all[instance_id]["smpl_betas"][
                            fi - self.start_timestep
                        ] = betas

                        body_pose = ins_smpl["smpl"]["body_pose"][fi]
                        smpl_orient = ins_smpl["smpl"]["global_orient"][fi]
                        cam_depend = ins_smpl["selected_cam_idx"][fi].item()

                        c2w = cam_to_worlds[cam_depend][fi - self.start_timestep]
                        world_orient = (
                            c2w[:3, :3].to(smpl_orient.device) @ smpl_orient.squeeze()
                        )
                        smpl_quats = matrix_to_quaternion(
                            torch.cat([world_orient[None, ...], body_pose], dim=0)
                        )

                        o2w = np.array(
                            instances_info[str(instance_id)]["frame_annotations"][
                                "obj_to_world"
                            ][ii]
                        )
                        o2w = torch.from_numpy(
                            np.linalg.inv(lidar_to_world_start) @ o2w
                        )
                        # box_size = instances_info[str(instance_id)]['frame_annotations']["box_size"][ii]

                        smpl_human_all[instance_id]["smpl_quats"][
                            fi - self.start_timestep
                        ] = smpl_quats
                        smpl_human_all[instance_id]["smpl_trans"][
                            fi - self.start_timestep
                        ] = o2w[:3, 3]
                        smpl_human_all[instance_id]["frame_valid"][
                            fi - self.start_timestep
                        ] = True

            self.smpl_human_all = smpl_human_all


class QcraftLiDARSource(SceneLidarSource):
    def __init__(
        self,
        lidar_data_config: OmegaConf,
        data_path: str,
        start_timestep: int,
        end_timestep: int,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(lidar_data_config, device=device)
        self.data_path = data_path
        self.start_timestep = start_timestep
        self.end_timestep = end_timestep
        self.create_all_filelist()
        self.load_data()

    def create_all_filelist(self):
        """
        Create a list of all the files in the dataset.
        e.g., a list of all the lidar scans in the dataset.
        """
        lidar_type = self.data_cfg.lidar_type
        assert lidar_type in ["lidar"]
        logger.info(f"Using '{lidar_type}' for lidar data.")

        lidar_filepaths = []
        # lidar_pose_filepaths = []
        for t in range(self.start_timestep, self.end_timestep):
            lidar_filepaths.append(
                os.path.join(self.data_path, lidar_type, "bin", f"{t:06d}.bin")
            )
            # lidar_pose_filepaths.append(
            #     os.path.join(self.data_path, "lidar_pose", f"{t:06d}.txt")
            # )
        self.lidar_filepaths = np.array(lidar_filepaths)
        # self.lidar_pose_filepaths = np.array(lidar_pose_filepaths)

    def load_calibrations(self):
        """
        Load the calibration files of the dataset.
        e.g., lidar to world transformation matrices.
        """
        lidar_to_worlds = []

        # we tranform the poses w.r.t. the first timestep to make the origin of the
        # first lidar pose as the origin of the world coordinate system.
        ### 将第一个时间步的 LiDAR 坐标作为世界坐标系的原点
        lidar_to_world_start = np.loadtxt(
            os.path.join(self.data_path, "lidar_pose", f"{self.start_timestep:06d}.txt")
        )
        for t in range(self.start_timestep, self.end_timestep):
            lidar_to_world_current = np.loadtxt(
                os.path.join(self.data_path, "lidar_pose", f"{t:06d}.txt")
            )
            # compute lidar_to_world transformation
            lidar_to_world = (
                np.linalg.inv(lidar_to_world_start) @ lidar_to_world_current
            )
            lidar_to_worlds.append(lidar_to_world)

        self.lidar_to_worlds = torch.from_numpy(
            np.stack(lidar_to_worlds, axis=0)
        ).float()

    def load_lidar(self):
        """
        Load the lidar data of the dataset from the filelist.
        """
        origins, directions, ranges, laser_ids = [], [], [], []
        timesteps = []

        accumulated_num_original_rays = 0
        accumulated_num_rays = 0
        for t in trange(
            0, len(self.lidar_filepaths), desc="Loading lidar", dynamic_ncols=True
        ):
            lidar_info = np.fromfile(self.lidar_filepaths[t], dtype=np.float32).reshape(
                -1, 5
            )

            # select lidar points based on the laser id
            if self.data_cfg.only_use_top_lidar:
                lidar_info = lidar_info[lidar_info[:, 4] == 0]

            original_length = len(lidar_info)
            accumulated_num_original_rays += original_length

            lidar_points = torch.from_numpy(lidar_info[:, :3]).float()
            lidar_origins = torch.zeros_like(lidar_points)

            lidar_origins = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_origins.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_points = (
                self.lidar_to_worlds[t][:3, :3] @ lidar_points.T
                + self.lidar_to_worlds[t][:3, 3:4]
            ).T
            lidar_ids = torch.from_numpy(lidar_info[:, 4]).float()
            lidar_directions = lidar_points - lidar_origins
            lidar_ranges = torch.norm(lidar_directions, dim=-1, keepdim=True)
            lidar_directions = lidar_directions / lidar_ranges
            lidar_timestamp = torch.ones_like(lidar_ranges).squeeze(-1) * t
            accumulated_num_rays += len(lidar_ranges)

            origins.append(lidar_origins)
            directions.append(lidar_directions)
            ranges.append(lidar_ranges)
            timesteps.append(lidar_timestamp)
            laser_ids.append(lidar_ids)

        logger.info(
            f"Number of lidar rays: {accumulated_num_rays} "
            f"({accumulated_num_rays / accumulated_num_original_rays * 100:.2f}% of "
            f"{accumulated_num_original_rays} original rays)"
        )

        self.origins = torch.cat(origins, dim=0)
        self.directions = torch.cat(directions, dim=0)
        self.ranges = torch.cat(ranges, dim=0)
        self.visible_masks = torch.zeros_like(self.ranges).squeeze().bool()
        self.colors = torch.ones_like(self.directions)
        self.laser_ids = torch.cat(laser_ids, dim=0)

        self._timesteps = torch.cat(timesteps, dim=0)
        self.register_normalized_timestamps()

    def to(self, device: torch.device):
        super().to(device)

    def get_lidar_rays(self, time_idx: int) -> Dict[str, Tensor]:
        """
        Get the of rays for rendering at the given timestep.
        Args:
            time_idx: the index of the lidar scan to render.
        Returns:
            a dict of the sampled rays.
        """
        origins = self.origins[self.timesteps == time_idx]
        directions = self.directions[self.timesteps == time_idx]
        ranges = self.ranges[self.timesteps == time_idx]
        normalized_time = self.normalized_time[self.timesteps == time_idx]
        return {
            "lidar_origins": origins,
            "lidar_viewdirs": directions,
            "lidar_ranges": ranges,
            "lidar_normed_time": normalized_time,
            "lidar_mask": self.timesteps == time_idx,
        }

    def delete_invisible_pts(self) -> None:
        if self.visible_masks is not None:
            num_bf = self.origins.shape[0]
            self.origins = self.origins[self.visible_masks]
            self.directions = self.directions[self.visible_masks]
            self.ranges = self.ranges[self.visible_masks]
            self._timesteps = self._timesteps[self.visible_masks]
            self._normalized_time = self._normalized_time[self.visible_masks]
            self.colors = self.colors[self.visible_masks]
            logger.info(
                f"[Lidar] {num_bf - self.visible_masks.sum()} out of {num_bf} points are cleared. {self.visible_masks.sum()} points left."
            )
            self.visible_masks = None
        else:
            logger.info("[Lidar] No unvisible points to clear.")
