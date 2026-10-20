#!/bin/bash
################################################################################
# 纯新轨迹渲染（不启用 DiFix）
#
# 使用方式：
#   run_dir=/path/to/run bash sim_render/cam/render_novel_trajectory.sh
#   默认读取 run_dir/group0 的 checkpoint 和 common.sh 中的 eval_group0_views。
#   结果写入 run_dir/novel_traj，布局中未选择的相机补黑。
#
# 轨迹配置：
#   novel_traj_shifts=left_1,left_2,right_1  # 便捷 token
#   traj_types_csv=original_traj,left_shift_1m  # 完整名称，设置后优先
################################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source scripts/qcraft/common.sh
source scripts/qcraft/difix_schedule.sh
source scripts/utils.sh

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ------------------------- 输入/输出 -------------------------
run_dir=${run_dir:-${1:-""}}
checkpoint_name=${checkpoint_name:-"checkpoint_final.pth"}
experiment_output_dir=${experiment_output_dir:-""}

# 默认只渲染 group0；可显式设置 render_groups=group0,group1 渲染两组。
render_groups=${render_groups:-"group0"}

# ------------------------- 轨迹 -------------------------
novel_traj_shifts=${novel_traj_shifts:-"left_3"}
traj_types_csv=${traj_types_csv:-""}
include_original_traj=${include_original_traj:-true}

# ------------------------- 渲染 -------------------------
fps=${fps:-10}
render_downscale=${render_downscale:-1}
render_rgb=${render_rgb:-true}
render_depth=${render_depth:-false}
save_images=${save_images:-true}
save_layout_video=${save_layout_video:-false}
save_full_layout_video=${save_full_layout_video:-true}
generate_lidar_pc=${generate_lidar_pc:-false}

# 完整分组 layout 依赖两组保存到同一目录下的逐帧图片。
if [[ "${save_full_layout_video}" == "true" ]]; then
    render_rgb=true
    save_images=true
fi

# 优先使用外部指定 GPU；未指定时自动选择。
render_gpu=${RENDER_GPU:-${pipeline_gpu:-}}
if [[ -z "${render_gpu}" ]]; then
    render_gpu=$(pick_gpu)
fi
if [[ -z "${render_gpu}" ]]; then
    echo "[ERROR] No available GPU found." >&2
    exit 1
fi

declare -a render_traj_types=()
if [[ -n "${traj_types_csv}" ]]; then
    IFS=',' read -r -a render_traj_types <<< "${traj_types_csv}"
else
    if [[ "${include_original_traj}" == "true" ]]; then
        render_traj_types+=("original_traj")
    fi
    IFS=',' read -r -a shift_tokens <<< "${novel_traj_shifts}"
    for token in "${shift_tokens[@]}"; do
        token="${token// /}"
        [[ -z "${token}" ]] && continue
        traj_type=$(_difix_shift_token_to_traj_type "${token}") || exit 1
        render_traj_types+=("${traj_type}")
    done
fi

if [[ ${#render_traj_types[@]} -eq 0 ]]; then
    echo "[ERROR] No trajectory type configured." >&2
    exit 1
fi

camera_csv_for_group() {
    case "$1" in
        group0|0)
            echo "${eval_group0_views}"
            ;;
        group1|1)
            echo "${eval_group1_views}"
            ;;
        *)
            echo "[ERROR] Unknown group '$1'. Expected group0 or group1." >&2
            return 1
            ;;
    esac
}

run_render() {
    local checkpoint="$1"
    local render_log_dir="$2"
    shift 2
    local -a render_cam_ids=("$@")
    local -a downscales=()
    local -a bool_args=()

    if [[ ! -f "${checkpoint}" ]]; then
        echo "[ERROR] Checkpoint not found: ${checkpoint}" >&2
        return 1
    fi
    if [[ ! -f "$(dirname "${checkpoint}")/config.yaml" ]]; then
        echo "[ERROR] Config not found next to checkpoint: $(dirname "${checkpoint}")/config.yaml" >&2
        return 1
    fi
    if [[ ${#render_cam_ids[@]} -eq 0 ]]; then
        echo "[ERROR] Empty camera list for checkpoint: ${checkpoint}" >&2
        return 1
    fi

    for _ in "${render_cam_ids[@]}"; do
        downscales+=("${render_downscale}")
    done

    [[ "${render_rgb}" == "true" ]] && bool_args+=(--render_rgb)
    [[ "${render_depth}" == "true" ]] && bool_args+=(--render_depth)
    [[ "${save_images}" == "true" ]] && bool_args+=(--save_images)
    [[ "${save_layout_video}" == "true" ]] && bool_args+=(--save_layout_video)
    [[ "${generate_lidar_pc}" == "true" ]] && bool_args+=(--generate_lidar_pc)

    mkdir -p "${render_log_dir}"
    echo "=========================================="
    echo "Rendering checkpoint: ${checkpoint}"
    echo "Output directory: ${render_log_dir}/novel_traj"
    echo "Camera IDs: ${render_cam_ids[*]}"
    echo "Trajectories: ${render_traj_types[*]}"
    echo "GPU: ${render_gpu}"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES="${render_gpu}" python sim_render/cam/render_novel_trajectory.py \
        --resume_from "${checkpoint}" \
        --traj_types "${render_traj_types[@]}" \
        --cam_ids "${render_cam_ids[@]}" \
        --downscales "${downscales[@]}" \
        --fps "${fps}" \
        "${bool_args[@]}" \
        "log_dir=${render_log_dir}"
}

if [[ -z "${run_dir}" ]]; then
    echo "[ERROR] Missing grouped run directory. Set run_dir or pass it as the first argument." >&2
    exit 1
fi

if [[ ! -d "${run_dir}" ]]; then
    echo "[ERROR] Run directory not found: ${run_dir}" >&2
    exit 1
fi

IFS=',' read -r -a groups <<< "${render_groups}"
if [[ ${#groups[@]} -eq 0 ]]; then
    echo "[ERROR] render_groups is empty." >&2
    exit 1
fi

output_root_dir=${experiment_output_dir:-"${run_dir}"}
render_view_order=""
for group in "${groups[@]}"; do
    group="${group// /}"
    [[ -z "${group}" ]] && continue
    group_cam_csv=$(camera_csv_for_group "${group}") || exit 1
    render_view_order="${render_view_order:+${render_view_order},}${group_cam_csv}"
    IFS=',' read -r -a group_cam_ids <<< "${group_cam_csv}"
    run_render \
        "${run_dir}/${group}/${checkpoint_name}" \
        "${output_root_dir}" \
        "${group_cam_ids[@]}"
done

if [[ "${save_full_layout_video}" == "true" ]]; then
    echo "[INFO] Merging grouped cameras into full layout videos."
    CUDA_VISIBLE_DEVICES="${render_gpu}" python tools/merge_grouped_novel_trajectory_layout.py \
        --novel_traj_dir "${output_root_dir}/novel_traj" \
        --traj_types "${render_traj_types[@]}" \
        --view_order "${render_view_order}" \
        --fps "${fps}" \
        --strict
fi

echo "[INFO] Novel-trajectory rendering completed."
