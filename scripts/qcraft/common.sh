#!/bin/bash
################################################################################
# 公共变量和函数
################################################################################


# 公共变量
################################################################################

# OBS过滤
OBS_FILTER='\[[=>_ \-]+\]|[0-9]+\.[0-9]+%|tps:|Succeed count|Failed count|Succeed bytes|Metrics|Task id| [0-9]+/[0-9]+ |[0-9]+\.[0-9]+KB/s|[0-9]+\.[0-9]+MB/[0-9]+\.[0-9]+MB'

# OBS配置
AK="${OBS_AK:-}"
SK="${OBS_SK:-}"
ENDPOINT="obs.cn-east-4.myhuaweicloud.com"
OBSUTIL="tools/obsutil_linux_amd64_5.8.3/obsutil"

# 数据根目录
raw_data_root="/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/raw/"
parsed_data_root="/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/raw/"
processed_data_root="${PROCESSED_DATA_ROOT:-/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/processed}"

# 训练配置
start_timestep=0
end_timestep=-1
gpu=0
segformer_path=/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/mmsegmentation
camera_ids=(0 1 2 3 5 6 7 9 10 11 12)
lidar_type="lidar" # lidar（运动补偿前）/visual（纯视觉）
output_root="${OUTPUT_ROOT:-/inspire/hdd/global_user/guoluosong-253108120129/chery_workspace/zhengzhehao/output}"
config_file="configs/omnire_extended_cam_lidar.yaml"
dataset_config="qcraft/11cams_${lidar_type}"
extra_config_info="road-hf-init" # 多视角颜色边缘引导的固定路面局部加密初始化

# 相机分组训练配置
group_execution_mode="serial"  # serial / parallel

train_group0_views=2,3,5,9,12
train_group1_views=0,1,6,7,10,11
eval_group0_views=2,3,5,9,12
eval_group1_views=0,1,6,7,10,11
num_view_groups=2

group0_gpu="${GROUP0_GPU:-0}"
group1_gpu="${GROUP1_GPU:-0}"

eval_downscale_when_loading="[1]"
view_order=0,1,2,3,5,6,7,9,10,11,12
fps=10

# LiDAR训练配置
train_lidar_iterations=3000
max_depth=100
block_size=50

# 公共函数
################################################################################

# 检查文件是否存在
check_file_exists() {
    local file_path="$1"
    local file_desc="${2:-文件}"
    
    if [ ! -f "$file_path" ]; then
        echo "=========================================="
        echo "✗ 错误: 未找到${file_desc}"
        echo "  Pipeline 已终止"
        echo "=========================================="
        return 1
    fi
    echo "=========================================="
    echo "✓ 步骤成功: 相机训练"
    echo "=========================================="
    return 0
}

# 检查目录是否为空
check_dir_not_empty() {
    local dir_path="$1"
    local dir_desc="${2:-目录}"
    
    if [ -z "$(ls -A "$dir_path" 2>/dev/null)" ]; then
        echo "=========================================="
        echo "✗ 错误: ${dir_desc}为空"
        echo "  Pipeline 已终止"
        echo "=========================================="
        return 1
    fi
    
    echo "=========================================="
    echo "✓ 步骤成功: LiDAR训练"
    echo "=========================================="
    return 0
}
