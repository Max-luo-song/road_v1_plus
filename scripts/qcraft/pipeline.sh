#!/bin/bash
################################################################################
# 完整Pipeline
################################################################################

# Preprocessing reads many camera images concurrently. Raise the per-process
# file descriptor limit for this pipeline when the container permits it.
OPEN_FILES_LIMIT=65535
if ! ulimit -Sn "${OPEN_FILES_LIMIT}"; then
    echo "[WARN] Unable to raise open-file limit to ${OPEN_FILES_LIMIT}; continuing with soft=$(ulimit -Sn), hard=$(ulimit -Hn)" >&2
fi
echo "[INFO] Open-file limit: $(ulimit -Sn)"

source scripts/qcraft/common.sh

scene_id="$1"
clip_start_time="${2:-0}"
clip_end_time="${3:-}"
# 默认跳过所有预处理，scene_id 直接填写 processed_data_root/training 下的目录名。
# 用法: bash scripts/qcraft/pipeline.sh <processed_scene_id>
# 显式传入 mode=0/1 仍可执行原来的完整预处理流程。
mode="${4:-2}"          # 0=本地 parsed 数据完整流程, 1=在线完整流程, 2=已有 processed 数据仅训练
label_type="${5:-pred}"
version="${6:-26.5.20.1}"
enable_difix="${7:-0}"   # 0=关闭, 1=开启 DiFix 渐进式蒸馏
novel_traj_shifts="${8:-left_1,left_2,left_3,right_1,right_2,right_3}"
novel_distill_repeat="${9:-3}"
original_distill_repeat="${10:-6}"

case "${mode}" in
    0|1|2)
        ;;
    *)
        echo "[ERROR] Unsupported mode: ${mode}" >&2
        echo "[ERROR] Expected: 0 (offline parsed), 1 (online), or 2 (processed-only training)" >&2
        exit 1
        ;;
esac

# 时间戳取整,start_time向下取整, end_time向上取整
clip_start_time=${clip_start_time%.*}
if [[ -n "$clip_end_time" && "$clip_end_time" == *.* ]]; then
    # 提取小数部分
    decimal=${clip_end_time##*.}
    # 如果小数部分 > 0，则 +1；否则保持不变
    if [[ "$decimal" != "0" && "$decimal" != "00" && "$decimal" != "000" ]]; then
        clip_end_time=$(( ${clip_end_time%.*} + 1 ))
    else
        clip_end_time=${clip_end_time%.*}
    fi
fi

if [[ "${mode}" == "0" || "${mode}" == "2" ]]; then
    scene_id_s_e="${scene_id}"
else
    scene_id_s_e=${scene_id}_${clip_start_time}_${clip_end_time}
fi
work_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
processed_scene_dir="${processed_data_root}/training/${scene_id_s_e}"

echo "参数: scene_id=$scene_id, clip_start_time=$clip_start_time, clip_end_time=$clip_end_time, mode=$mode, label_type=$label_type, version=$version, enable_difix=$enable_difix, novel_traj_shifts=$novel_traj_shifts"
echo "Processed data root: ${processed_data_root}"
echo "Output root: ${output_root}"
if [[ "${enable_difix}" != "1" ]]; then
    echo "[INFO] enable_difix=0, novel_traj_shifts ignored"
fi

if [[ "${mode}" == "2" ]]; then
    echo "=========================================="
    echo "✓ Mode 2: 使用已有 processed 数据，仅执行训练"
    echo "  输入目录: ${processed_scene_dir}"
    echo "  输出根目录: ${output_root}"
    echo "=========================================="

    if [[ ! -d "${processed_scene_dir}" ]]; then
        echo "[ERROR] Processed scene directory not found: ${processed_scene_dir}" >&2
        exit 1
    fi
    if [[ -z "$(ls -A "${processed_scene_dir}" 2>/dev/null)" ]]; then
        echo "[ERROR] Processed scene directory is empty: ${processed_scene_dir}" >&2
        exit 1
    fi

    mkdir -p "${output_root}"
    if [[ ! -w "${output_root}" ]]; then
        echo "[ERROR] Output root is not writable: ${output_root}" >&2
        exit 1
    fi

    echo "[INFO] Skipping data download/parsing, label extraction, preprocessing, mask extraction, and human-pose processing."
else
if [[ "${mode}" == "0" ]]; then
    abs_data_root=$(realpath "${parsed_data_root}")
else
    abs_data_root=$(realpath "${parsed_data_root}/${scene_id}")
fi

# 在线/本地 parsed 完整流程可能会调用 OBS 工具，提前确保其可执行。
chmod -R 755 tools/obsutil_linux_amd64_5.8.3

# step 1: 数据下载切分和解析
echo "=========================================="
echo "✓ 开始数据下载切分和解析"
echo "=========================================="
bash scripts/qcraft/data_preparation.sh "$scene_id" "$clip_start_time" "$clip_end_time" "$mode"
echo "=========================================="
echo "✓ 步骤结束: 数据下载切分和解析"
echo "=========================================="

# step 2: extract labels
echo "=========================================="
echo "✓ 开始标签提取"
echo "=========================================="
if [[ "${mode}" == "0" ]]; then
    echo "[INFO] Offline parsed mode: using existing labels in ${abs_data_root}/${scene_id_s_e}/label"
else
    bash scripts/qcraft/label_extraction.sh "$scene_id" "$clip_start_time" "$clip_end_time" "$label_type" "$version"
fi
echo "=========================================="
echo "✓ 步骤结束: 标签提取"
echo "=========================================="

# step 3: preprocess
echo "=========================================="
echo "✓ 开始预处理"
echo "=========================================="
#source /opt/conda/etc/profile.d/conda.sh && conda activate main
export PYTHONPATH=$(pwd)
python datasets/preprocess.py \
    --data_root ${abs_data_root} \
    --target_dir ${processed_data_root} \
    --dataset qcraft \
    --split training \
    --scene_ids $scene_id_s_e \
    --workers 2 \
    --process_keys ego_masks images calib pose objects dynamic_masks lidar \
    --skip_front_wide_side_cameras \
    --label_type ${label_type}
echo "=========================================="
echo "✓ 步骤结束: 预处理"
echo "=========================================="

# step 4: extract mask
echo "=========================================="
echo "✓ 开始掩码提取"
echo "=========================================="
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${SCRIPT_DIR}/utils.sh
conda activate mmseg2

# Pick an avaliable gpu
source scripts/utils.sh
if [ "${gpu}" = "-1" ]; then
    gpu=$(pick_gpu)
    if [ -z "${gpu}" ]; then
        echo "no gpu found"
        exit 1
    fi
fi
echo "Using GPU: ${gpu}"

CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/extract_masks_mmseg2.py \
    --data_root ${processed_data_root}/training \
    --config=${segformer_path}/configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py \
    --checkpoint=${segformer_path}/checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth \
    --scene_ids=${scene_id_s_e} \
    --process_dynamic_mask
echo "=========================================="
echo "✓ 步骤结束: 掩码提取"
echo "=========================================="

# step 5: process human pose
echo "=========================================="
echo "✓ 开始处理人体姿态"
echo "=========================================="
conda activate 4D-humans
export PYTHONPATH=$PYTHONPATH:${work_root}/third_party/Humans4D
CUDA_VISIBLE_DEVICES=${gpu} python datasets/tools/humanpose_process.py \
    --data_root ${processed_data_root}/training \
    --dataset qcraft \
    --scene_id $scene_id_s_e
echo "=========================================="
echo "✓ 步骤结束: 处理人体姿态"
echo "=========================================="
fi

# step 6: camera group training
echo "=========================================="
echo "✓ 开始相机训练（仅 group0）"
echo "=========================================="

run_dir="$(bash "${work_root}/scripts/qcraft/group_training.sh" "${scene_id_s_e}" group0)" || exit 1

echo "Run directory: ${run_dir}"

check_file_exists "${run_dir}/group0/checkpoint_final.pth" "group0 checkpoint_final.pth" || exit 1

echo "=========================================="
echo "✓ 步骤结束: 相机分组训练"
echo "=========================================="

# step 7: lidar训练
echo "=========================================="
echo "✓ 开始LiDAR训练"
echo "=========================================="
source /opt/conda/etc/profile.d/conda.sh && conda activate main
LIDAR_PROJECT_ROOT="${work_root}/lidargs"
echo "LiDAR project root: ${LIDAR_PROJECT_ROOT}"
export PYTHONPATH="${LIDAR_PROJECT_ROOT}:${PYTHONPATH}"

echo "Data path: ${processed_scene_dir}"

lidar_cpt_path="${run_dir}/lidar_cpt"

mkdir -p "$lidar_cpt_path"
echo "LiDAR checkpoint path: ${lidar_cpt_path}"

python ${LIDAR_PROJECT_ROOT}/train.py \
    -s "${processed_scene_dir}" \
    -m ${lidar_cpt_path} \
    --caseid ${scene_id_s_e} \
    --gpu ${gpu} \
    --iterations ${train_lidar_iterations} \
    --max_depth ${max_depth} \
    --dataset chery \
    --block_size ${block_size}
check_dir_not_empty "$lidar_cpt_path" "lidar_cpt" || exit 1

# step 8: DiFix progressive distillation（enable_difix=1 时执行）
if [[ "${enable_difix}" == "1" ]]; then
    echo "=========================================="
    echo "✓ 开始 DiFix 渐进式蒸馏（仅 group0）"
    echo "  novel_traj_shifts: ${novel_traj_shifts}"
    echo "  output_dir: ${run_dir}"
    echo "=========================================="

    group_0_difix_ckpt_path="${run_dir}/group0/checkpoint_final.pth"
    group_0_original_view_ckpt_path="${run_dir}/group0/checkpoint_final_original_view.pth"

    # If an original-view backup exists, restore it before DiFix so repeated runs
    # always start from the non-distilled model.
    if [ -f "${group_0_original_view_ckpt_path}" ]; then
        echo "[INFO] Restoring group0 original-view checkpoint to checkpoint_final.pth"
        rm -f "${group_0_difix_ckpt_path}" || exit 1
        cp -p "${group_0_original_view_ckpt_path}" "${group_0_difix_ckpt_path}" || exit 1
    fi

    if [ ! -f "${group_0_difix_ckpt_path}" ]; then
        echo "[ERROR] group0 3DGS checkpoint not found"
        echo "  expected: ${group_0_difix_ckpt_path}"
        exit 1
    fi

    # 与 DiFix 脚本使用相同的规则，提前确定蒸馏产物路径。
    source scripts/qcraft/difix_schedule.sh
    difix_ckpt_suffix=$(build_difix_ckpt_suffix \
        "${novel_traj_shifts}" \
        "${novel_distill_repeat}" \
        "${original_distill_repeat}") || exit 1
    group_0_difix_final_ckpt_path="${run_dir}/group0/checkpoint_final_${difix_ckpt_suffix}.pth"

    source /opt/conda/etc/profile.d/conda.sh && conda activate main
    cd "${work_root}"
    export PYTHONPATH=$(pwd)

    # group 0 difix training
    env \
        ckpt_path="${group_0_difix_ckpt_path}" \
        experiment_output_dir="${run_dir}/group0" \
        difix_group=group0 \
        novel_traj_shifts="${novel_traj_shifts}" \
        novel_distill_repeat="${novel_distill_repeat}" \
        original_distill_repeat="${original_distill_repeat}" \
        pipeline_gpu="${gpu}" \
        bash sim_render/cam/render_novel_trajectory_with_difix.sh

    if [ $? -ne 0 ]; then
        echo "[ERROR] group 0 DiFix distillation failed"
        exit 1
    fi

    if [ ! -f "${group_0_difix_final_ckpt_path}" ]; then
        echo "[ERROR] DiFix final checkpoint not found:"
        echo "  ${group_0_difix_final_ckpt_path}"
        exit 1
    fi

    if [ ! -f "${group_0_original_view_ckpt_path}" ]; then
        cp -p "${group_0_difix_ckpt_path}" "${group_0_original_view_ckpt_path}" || exit 1
        echo "[INFO] Original checkpoint backed up to: ${group_0_original_view_ckpt_path}"
    else
        echo "[INFO] Keep existing original-view checkpoint: ${group_0_original_view_ckpt_path}"
    fi
    rm -f "${group_0_difix_ckpt_path}" || exit 1
    cp -f "${group_0_difix_final_ckpt_path}" "${group_0_difix_ckpt_path}" || exit 1
    echo "[INFO] DiFix checkpoint promoted to: ${group_0_difix_ckpt_path}"

    echo "=========================================="
    echo "✓ 步骤结束: DiFix 渐进式蒸馏"
    echo "=========================================="
fi

# step 9: 仅在线 mode=1 上传数据。
if [[ "${mode}" != "1" ]]; then
    echo "=========================================="
    echo "✓ Mode ${mode}: 跳过上传"
    echo "  本地训练结果: ${run_dir}"
    echo "=========================================="
else
    echo "=========================================="
    echo "✓ 开始上传数据"
    echo "=========================================="
    bash scripts/qcraft/upload.sh "$scene_id" "$clip_start_time" "$clip_end_time" "$version" "$run_dir"
fi
