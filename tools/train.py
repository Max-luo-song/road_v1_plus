from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging_utils import MetricLogger, setup_logging
from utils.view_parallel_utils import apply_view_parallel_config, log_view_parallel_status
from models.trainers.base import BasicTrainer
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset, DepthMode
from scipy.spatial import ConvexHull
from matplotlib.path import Path

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

@torch.no_grad()
def clean_road_overhead_during_training(trainer, road_model, height_limit=0.5, grid_spacing=0.5, below_road_margin=0.5):
    bg_model = trainer.models['Background']
    bg_xyz = bg_model._means
    bg_scales = bg_model.get_scaling
    road_xyz = road_model._means
    if road_xyz.numel() == 0:
        print("Road model has no points; skipping road-based geometric pruning.")
        return

    # 1. 根据路面范围计算凸包
    road_xy_np = road_xyz[:, :2].cpu().numpy()
    hull = ConvexHull(road_xy_np)
    # 2.获取凸包顶点并向外扩张 0.2m (Padding)
    vertices = road_xy_np[hull.vertices]
    center = np.mean(vertices, axis=0)
    expanded_vertices = vertices + (vertices - center) * 0.15 # 略微等比例外扩
    hull_path = Path(expanded_vertices)

    bg_xy_np = bg_xyz[:, :2].cpu().numpy()
    mask_in_hull = torch.from_numpy(hull_path.contains_points(bg_xy_np)).to(bg_xyz.device)
    
    if not mask_in_hull.any():
        print("without mask_in_hull! Skipping hull-based pruning, global depth pruning only.")

    # 增加网格尺寸，使查找更容易匹配
    x_min, y_min = road_xyz[:, :2].min(dim=0)[0] - 1.0
    x_max, y_max = road_xyz[:, :2].max(dim=0)[0] + 1.0
    
    nx = int((x_max - x_min) / grid_spacing) + 1
    ny = int((y_max - y_min) / grid_spacing) + 1
    
    # 使用 amax 记录最高路面点，保证判定上限
    height_map = torch.full((nx, ny), -999.0, device=bg_xyz.device)
    road_ix = ((road_xyz[:, 0] - x_min) / grid_spacing).long().clamp(0, nx - 1)
    road_iy = ((road_xyz[:, 1] - y_min) / grid_spacing).long().clamp(0, ny - 1)
    height_map.view(-1).scatter_reduce_(0, road_iy * nx + road_ix, road_xyz[:, 2], reduce="amax", include_self=False)

    # 3. 查找与判定（仅凸包内）
    full_kill_mask = torch.zeros(bg_xyz.shape[0], dtype=torch.bool, device=bg_xyz.device)
    global_road_z = torch.median(road_xyz[:, 2])

    if mask_in_hull.any():
        relevant_bg_indices = torch.where(mask_in_hull)[0]
        rel_bg_xyz = bg_xyz[relevant_bg_indices]
        rel_bg_scales = bg_scales[relevant_bg_indices]

        bg_ix = ((rel_bg_xyz[:, 0] - x_min) / grid_spacing).long().clamp(0, nx - 1)
        bg_iy = ((rel_bg_xyz[:, 1] - y_min) / grid_spacing).long().clamp(0, ny - 1)
        
        local_road_z = height_map[bg_ix, bg_iy]

        # 只要格子附近有路面 (valid_road_mask) 或者回退到全局路面平均高度
        valid_road_mask = local_road_z > -900.0
        
        # 最终使用的参考高度：优先局部，局部没有则使用全局
        ref_z = torch.where(valid_road_mask, local_road_z, global_road_z)

        # 4. 删除条件：1）中心点在路面参考高度 + height_limit 以下，2）高斯的底部触及路面
        bg_max_scale = rel_bg_scales.max(dim=-1)[0]
        bg_bottom_z = rel_bg_xyz[:, 2] - bg_max_scale * 2.0 # 稍微收敛到 2 sigma
        
        # 条件 1: 中心点在路面限高以下 (rel_bg_xyz[:, 2] < ref_z + height_limit)
        # 条件 2: 或者是大高斯的底部触及了路面限高 (bg_bottom_z < ref_z + height_limit)
        local_kill_mask = (rel_bg_xyz[:, 2] < ref_z + height_limit) | \
                            (bg_bottom_z < ref_z + height_limit)

        # 5. 凸包区域剔除 mask
        full_kill_mask[relevant_bg_indices] = local_kill_mask

    # 6. 全局深度裁剪：所有背景高斯中，Z 低于路面中值 - below_road_margin 的一律删除
    below_road_mask = bg_xyz[:, 2] < global_road_z - below_road_margin
    num_below = below_road_mask.sum().item()
    if num_below > 0:
        print(f"--- [路面深度裁剪] 剔除 {num_below} 个路面以下背景高斯 "
              f"(margin: {below_road_margin}m, road_z: {global_road_z:.2f}) ---")
    full_kill_mask = full_kill_mask | below_road_mask

    if full_kill_mask.any():
        num_to_kill = full_kill_mask.sum().item()
        print(f"--- [强制清理] 共剔除 {num_to_kill} 个背景高斯 (hull_limit: {height_limit}m, below_margin: {below_road_margin}m) ---")
        bg_model.prune_points(full_kill_mask, optimizer=trainer.optimizer)

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    apply_view_parallel_config(cfg, args, append_group_to_log_dir=True)
    log_dir = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./", 
        ["configs", "datasets", "models", "utils", "tools"], 
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data)
    log_view_parallel_status(cfg, dataset)

    # setup trainer
    trainer: BasicTrainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.aabb.reshape(2, 3),
        device=device
    )
    if trainer.depth_loss_fn is not None:
        # 仅在 static 模式下 mask 掉动态区域
        mask_out_dynamic_regions = dataset.depth_mode != DepthMode.SINGLE_FRAME_RAW
        if mask_out_dynamic_regions:
            logger.info("Dynamic region masking is enabled in depth loss.")
        else:
            logger.info("Dynamic region masking is disabled in depth loss.")

        trainer.depth_loss_fn.set_dynamic_region_masking(mask_out_dynamic_regions)

    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(
            dataset=dataset, initialization_output_dir=os.path.join(cfg.log_dir, "road_init"))
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [   ## 体现在output里面的images中，每隔几k就会输出
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
        "gt_road_rgbs",
        "road_rgbs"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    # setup optimizer  
    trainer.initialize_optimizer()
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    
    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    # NOTE(gls): Geometric Pruning for background on road prepare
    if 'RoadNodes' in trainer.gaussian_classes:
        road_model = trainer.models['RoadNodes']
        road_xyz = road_model._means.detach()
        if road_xyz.numel() == 0:
            road_stats = None
            print("Warning: Road model has no points, skipping geometric pruning setup.")
        else:
            # 预计算路面包围盒信息
            road_stats = {
                'min_x': road_xyz[:, 0].min().item(),
                'max_x': road_xyz[:, 0].max().item(),
                'min_y': road_xyz[:, 1].min().item(),
                'max_y': road_xyz[:, 1].max().item(),
                'avg_z': road_xyz[:, 2].mean().item() # 假设路面大致平坦，取平均高度；如果有坡度，后续逻辑需要改为 KNN 
            }
            print(f"Road Geometry Loaded: Z~={road_stats['avg_z']:.2f}, X[{road_stats['min_x']:.1f}, {road_stats['max_x']:.1f}]")
    else:
        road_stats = None
        print("Warning: Road model not found, skipping geometric pruning setup.")

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]

            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                )

            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                    }
                )

            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()
                
        
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad() # zero grad
        
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        
        # forward & backward
        # drivestudio/models/trainers/scene_graph.py
        outputs, gs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
            gs=gs 
        )
        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        trainer.backward(loss_dict)
        
        # after training step
        trainer.postprocess_per_train_step(step=step)
        
        # NOTE(gls)：Geometric Pruning for background on road 
        # if 'RoadNodes' in trainer.gaussian_classes and 'Background' in trainer.gaussian_classes:
        #     if step > 500 and step % 100 == 0:  #TODO(gls):100变成config超参
        #         clean_road_overhead_during_training(
        #                 trainer = trainer,
        #                 road_model = trainer.models['RoadNodes'], # 确保 key 与你注册模型时一致
        #                 height_limit = 0.5, 
        #                 grid_spacing=0.15,
        #                 below_road_margin=0.5
        #             )
        #         torch.cuda.empty_cache() # 清理完后清理显存，防止碎片化
        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) and (args.resume_from is None)
        if do_save:  # 保存之前再做一遍筛选
            # if 'RoadNodes' in trainer.gaussian_classes and 'Background' in trainer.gaussian_classes:
            #     clean_road_overhead_during_training(
            #             trainer = trainer,
            #             road_model = trainer.models['RoadNodes'],
            #             height_limit = 0.5, 
            #             grid_spacing=0.15,
            #             below_road_margin=0.5
            #         )    
            #     torch.cuda.empty_cache()
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )
        
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")
            
    
    logger.info("Training done!")

    if not cfg.get("view_parallel", {}).get("enabled", False) or args.run_final_eval:
        do_evaluation(
            step=step,
            cfg=cfg,
            trainer=trainer,
            dataset=dataset,
            render_keys=render_keys,
            args=args,
        )
    else:
        logger.info("Skipping final per-group evaluation in view-parallel training.")
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    parser.add_argument("--run_final_eval", action="store_true", help="run final evaluation after training even in view-parallel mode")
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")

    # view-parallel training
    parser.add_argument("--view_parallel", action="store_true", help="enable view-parallel mode")
    parser.add_argument("--num_view_groups", type=int, default=2, help="number of view groups")
    parser.add_argument("--view_group_id", type=int, default=None, help="view group id for this process")
    parser.add_argument("--view_ids", type=str, default=None, help="comma-separated camera ids for this process")
    parser.add_argument("--gpu_id", type=int, default=None, help="physical gpu id used by this process, for logging")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    final_step = main(args)
