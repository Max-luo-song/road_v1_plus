"""Load explicit marking masks, or identify candidates in observed road RGB."""
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils.lane_masks import image_lane_candidates

logger = logging.getLogger(__name__)


@torch.no_grad()
def attach_lane_masks(camera_data, config):
    source = config.get("lane_mask_source", "auto")
    if source not in ("auto", "labels", "image"):
        raise ValueError("lane_mask_source must be auto, labels or image")
    for camera in camera_data.values():
        if camera.road_masks is None:
            raise ValueError("lane supervision requires load_road_mask=true")
        masks, inferred_frames, pixels = [], 0, 0
        root = Path(config.get("lane_mask_dir") or Path(camera.data_path) / "lane_masks")
        for i in range(camera.num_frames):
            path = root / f"{camera.start_timestep+i:06d}_{camera.cam_id}.png"
            if source != "image" and path.is_file():
                mask = np.array(Image.open(path).convert("L").resize((camera.WIDTH, camera.HEIGHT), Image.NEAREST))
                if camera.undistort:
                    import cv2
                    mask = cv2.undistort(mask, camera.intrinsics[i].cpu().numpy(), camera.distortions[i].cpu().numpy())
                mask = torch.from_numpy(mask > 0).to(camera.images.device)
            elif source == "labels":
                raise FileNotFoundError(f"Required marking mask not found: {path}")
            else:
                inferred_frames += 1
                mask = image_lane_candidates(
                    camera.images[i], camera.road_masks[i],
                    min_brightness=float(config.get("lane_min_brightness", 0.48)),
                    min_contrast=float(config.get("lane_min_contrast", 0.06)),
                    window_size=int(config.get("lane_contrast_window", 15)),
                    white_chroma=float(config.get("lane_white_chroma", 0.22)),
                )
            mask &= camera.road_masks[i].bool()
            for occlusion in (camera.egocar_mask, getattr(camera, "dynamic_masks", None),
                              getattr(camera, "human_masks", None), getattr(camera, "vehicle_masks", None)):
                if occlusion is not None:
                    frame_mask = occlusion[i] if occlusion.ndim == 3 else occlusion
                    mask &= ~frame_mask.bool()
            pixels += int(mask.sum())
            masks.append(mask)
        camera.lane_masks = torch.stack(masks)
        logger.info("Marking masks camera=%s frames=%d candidate_pixels=%d image_inferred_frames=%d",
                    camera.cam_id, len(masks), pixels, inferred_frames)
        if inferred_frames:
            logger.warning("Camera %s uses RGB marking candidates for %d frames; these are not annotated lane labels.",
                           camera.cam_id, inferred_frames)
        if pixels == 0:
            logger.warning("Camera %s has no marking supervision pixels; check lane masks/thresholds.", camera.cam_id)
