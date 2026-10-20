from pathlib import Path
from types import SimpleNamespace
import json, logging
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from utils.road_high_frequency import initialize_high_frequency_road, project_points
logging.basicConfig(level=logging.INFO)
torch.set_num_threads(4)
base=Path('/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao')
data=base/'processed/training/20260613_102138_QCJPTDJ07384_3298_3308'
ckpath=base/'output/qcraft_20260613_102138_QCJPTDJ07384_3298_3308/20261009_lidar+traincam2_3_5_9_12_0_1_6_7_10_11road-color-coverage/group0/checkpoint_final.pth'
frames=[30,40,50];cameras={}
origin=np.loadtxt(data/'lidar_pose/000000.txt')
for cam_id in [2,3,5,9,12]:
 images=[];roads=[];dynamic=[];sky=[];poses=[]
 intr=np.loadtxt(data/f'intrinsics/{cam_id}.txt')
 K=torch.tensor([[intr[0],0,intr[2]],[0,intr[1],intr[3]],[0,0,1]],dtype=torch.float32)
 ext=np.loadtxt(data/f'extrinsics/{cam_id}.txt')
 def loadmask(p):return torch.from_numpy(np.array(Image.open(p).convert('L'))>0)
 for f in frames:
  p=next((data/'images').glob(f'{f:06d}_{cam_id}.*'))
  images.append(torch.from_numpy(np.array(Image.open(p).convert('RGB'))).float()/255)
  roads.append(loadmask(data/f'road_masks/{f:06d}_{cam_id}.png'))
  dynamic.append(loadmask(data/f'fine_dynamic_masks/all/{f:06d}_{cam_id}.png'))
  sky.append(loadmask(data/f'sky_masks/{f:06d}_{cam_id}.png'))
  poses.append(torch.tensor(np.linalg.inv(origin)@np.loadtxt(data/f'lidar_pose/{f:06d}.txt')@ext,dtype=torch.float32))
 cameras[cam_id]=SimpleNamespace(images=torch.stack(images),road_masks=torch.stack(roads),dynamic_masks=torch.stack(dynamic),sky_masks=torch.stack(sky),egocar_mask=loadmask(data/f'ego_masks/{cam_id}.png'),cam_to_worlds=torch.stack(poses),intrinsics=K.repeat(len(frames),1,1),lidar_depth_maps=None)
ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True)
means=ck['models']['RoadNodes']['_means']
cfg=OmegaConf.to_container(OmegaConf.load('configs/omnire_extended_cam_lidar.yaml').model.RoadNodes.init.high_frequency)
cfg.update(frame_stride=1,max_views=15)
pix,depth,inside=project_points(means,cameras[2],1,cfg)
valid=inside&(depth>5)&(depth<35)&cameras[2].road_masks[1].flatten()[pix]
center=means[valid,:2].median(0).values
radius=8.
while True:
 keep=((means[:,:2]-center).abs()<=radius).all(-1)
 if int(keep.sum())<=100000:break
 radius*=.8
points=means[keep].clone()
out=Path('docs/road_hf_preview')
added,colors,stats=initialize_high_frequency_road(points,torch.full_like(points,.4),cameras,[0,1,2],cfg,out)
context=dict(source_checkpoint=str(ckpath),source_frames=frames,source_cameras=list(cameras),base_patch_radius=radius,base_patch_center=center.tolist(),limitation='CPU local-patch evidence/sampling preview using frozen checkpoint centers, not a fresh full-scene initialization or CUDA training; sparse LiDAR depth unavailable in this preview.')
(out/'context.json').write_text(json.dumps(context,indent=2))
print(json.dumps(stats,indent=2),flush=True)
