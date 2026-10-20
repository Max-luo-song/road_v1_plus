import json, shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw

cv2.setNumThreads(2)
base=Path('/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao')
root=base/'output/qcraft_20260613_102138_QCJPTDJ07384_3298_3308'
runs={'old':root/'20261009_lidar+traincam2_3_5_9_12_0_1_6_7_10_11road-color-coverage','new':root/'20261019_lidar+traincam2_3_5_9_12road-adaptive-s-o'}
data=base/'processed/training/20260613_102138_QCJPTDJ07384_3298_3308'
out=Path(__file__).resolve().parent

metrics={name:json.loads(next((run/'group0/metrics_eval').glob('*.json')).read_text()) for name,run in runs.items()}
counts={};sums={name:{cam:np.zeros(6,dtype=np.float64) for cam in [2,3,5,9,12]} for name in runs}
for cam in [2,3,5,9,12]:
 paths=[{p.parent.name:p for p in (run/'novel_traj/original_traj_step40000/images').glob(f'*/{cam}_rgbs.png')} for run in runs.values()]
 common=sorted(set(paths[0])&set(paths[1]));counts[cam]=len(common)
 for frame in common:
  gtpath=next((data/'images').glob(f'{frame}_{cam}.*'))
  gt=np.asarray(Image.open(gtpath).convert('RGB'),dtype=np.float32)/255
  h,w=gt.shape[:2]
  def mask(path):return np.asarray(Image.open(path).convert('L').resize((w,h),Image.Resampling.NEAREST))>0
  road=mask(data/f'road_masks/{frame}_{cam}.png')
  support=road.copy()
  for p in [data/f'dynamic_masks/{frame}_{cam}.png',data/f'ego_masks/{cam}.png',data/f'sky_masks/{frame}_{cam}.png']:
   if p.exists():road &= ~mask(p)
  road=cv2.erode(road.astype('uint8'),np.ones((7,7),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
  lum=gt @ np.array([.2126,.7152,.0722],dtype=np.float32)
  local=cv2.boxFilter(lum*support,-1,(15,15),borderType=cv2.BORDER_CONSTANT)/np.maximum(cv2.boxFilter(support.astype('float32'),-1,(15,15),borderType=cv2.BORDER_CONSTANT),1e-6)
  white=np.ptp(gt,axis=-1)<=.22
  yellow=(gt[:,:,0]>=.45)&(gt[:,:,1]>=.4)&(gt[:,:,2]<.75*np.minimum(gt[:,:,0],gt[:,:,1]))
  lane=road&(white|yellow)&(lum>=.48)&(lum-local>=.06)
  for name,p in zip(runs,[paths[0][frame],paths[1][frame]]):
   pred=np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255
   assert pred.shape==gt.shape,(p,pred.shape,gt.shape)
   err=np.abs(pred-gt).mean(-1)
   edge_sum=0.;edge_count=0
   for axis in [0,1]:
    a=(slice(1,None),slice(None)) if axis==0 else (slice(None),slice(1,None))
    b=(slice(None,-1),slice(None)) if axis==0 else (slice(None),slice(None,-1))
    valid=road[a]&road[b]&(lane[a]|lane[b])
    edge_sum+=np.abs((pred[a]-pred[b])-(gt[a]-gt[b])).mean(-1)[valid].sum()
    edge_count+=valid.sum()
   sums[name][cam]+=np.array([err[road].sum(),road.sum(),err[lane].sum(),lane.sum(),edge_sum,edge_count])
 print('processed cam',cam,'frames',len(common),flush=True)
summary={}
for name in runs:
 summary[name]={}
 for cam in [2,3,5,9,12,'all']:
  s=sums[name][cam] if cam!='all' else np.stack(list(sums[name].values())).sum(0)
  summary[name][str(cam)]={'road_rgb_mae':float(s[0]/s[1]),'lane_candidate_rgb_mae':float(s[2]/s[3]),'lane_candidate_edge_mae':float(s[4]/s[5]),'road_pixels':int(s[1]),'lane_candidate_pixels':int(s[3])}
payload={'runs':{k:str(v) for k,v in runs.items()},'paired_original_frames_per_camera':counts,'group0_eval':metrics,'original_trajectory_masked_metrics':summary,'checkpoint':json.loads((out/'checkpoint_stats.json').read_text()),'limitations':['group0 only; old merged 11-camera metrics are not used','original-trajectory masks from processed road/dynamic/ego/sky; eroded 3px','lane masks are RGB candidates, not annotated labels','novel trajectory has no target RGB or alpha outputs; visual black patches cannot be identified as transparency holes from RGB alone']}
(out/'measurements.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(summary,indent=2))
