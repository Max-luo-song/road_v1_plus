#!/bin/bash
################################################################################
# 相机分组训练、评估和结果合并
################################################################################
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"
source scripts/qcraft/common.sh

scene_id_s_e="$1"
selected_group="${2:-group0}"
case "${selected_group}" in
    all|group0) ;;
    *) echo "[ERROR] Expected group selection: all or group0" >&2; exit 1 ;;
esac

project_name="qcraft_${scene_id_s_e}"
date_str="$(date +%Y%m%d)"
train_view_tag="${train_group0_views//,/_}"
if [[ "${selected_group}" == "all" ]]; then
    train_view_tag+="_${train_group1_views//,/_}"
fi
if [[ -n "${RUN_NAME_OVERRIDE:-}" ]]; then
    if [[ ! "${RUN_NAME_OVERRIDE}" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]]; then
        echo "[ERROR] Invalid RUN_NAME_OVERRIDE: ${RUN_NAME_OVERRIDE}" >&2
        echo "[ERROR] Use only letters, numbers, '.', '_', '+', and '-'." >&2
        exit 1
    fi
    run_name="${RUN_NAME_OVERRIDE}"
else
    run_name="${date_str}_${lidar_type}+traincam${train_view_tag}${extra_config_info}"
fi
run_dir="${output_root}/${project_name}/${run_name}"

log() {
    echo "$@" >&2
}

log "[INFO] Training config: ${config_file}"

if grep -qE '^[[:space:]]+lane_influence_radius:' "${config_file}"; then
    processed_scene_dir="${processed_data_root}/training/${scene_id_s_e}"
    bevrg_label_dir="${processed_scene_dir}/label/bev_road_geometry_label/bevrg_label_20260320"
    if [[ ! -d "${bevrg_label_dir}" ]]; then
        log "[ERROR] BEVRG label directory not found: ${bevrg_label_dir}"
        exit 1
    fi
    bevrg_label_count="$(find "${bevrg_label_dir}" -maxdepth 1 -type f -name '*.json' | wc -l)"
    if [[ "${bevrg_label_count}" -eq 0 ]]; then
        log "[ERROR] No JSON labels found in ${bevrg_label_dir}"
        exit 1
    fi
    log "[INFO] BEVRG labels ready: ${bevrg_label_dir} (${bevrg_label_count} JSON files)"
fi

case "${group_execution_mode}" in
    serial|parallel)
        ;;
    *)
        log "[ERROR] Unsupported group_execution_mode: ${group_execution_mode}"
        log "[ERROR] Expected: serial or parallel"
        exit 1
        ;;
esac

pid0=""
pid1=""

cleanup_children() {
    local reason="${1:-EXIT}"

    trap - INT TERM EXIT

    log
    log "Caught ${reason}, cleaning up child process groups..."

    for pid in "${pid0:-}" "${pid1:-}"; do
        if [ -z "${pid}" ]; then
            continue
        fi

        if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
            log "Stopping process group/PID ${pid}"
            kill -- "-${pid}" 2>/dev/null \
                || kill "${pid}" 2>/dev/null \
                || true
        fi
    done

    sleep 3

    for pid in "${pid0:-}" "${pid1:-}"; do
        if [ -z "${pid}" ]; then
            continue
        fi

        if kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null; then
            log "Force killing process group/PID ${pid}"
            kill -9 -- "-${pid}" 2>/dev/null \
                || kill -9 "${pid}" 2>/dev/null \
                || true
        fi
    done
}

show_log_tail() {
    local log_file="$1"

    if [ -n "${log_file}" ] && [ -f "${log_file}" ]; then
        log "===== ${log_file} tail ====="
        tail -n 80 "${log_file}" >&2 || true
    fi
}

start_train_group() {
    local group_id="$1"
    local physical_gpu="$2"
    local view_ids="$3"
    local log_file="$4"
    local pid_var="$5"

    setsid env CUDA_VISIBLE_DEVICES="${physical_gpu}" python -u tools/train.py \
        --config_file "${config_file}" \
        --output_root "${output_root}" \
        --project "${project_name}" \
        --run_name "${run_name}" \
        --view_group_id "${group_id}" \
        --view_ids "${view_ids}" \
        --gpu_id 0 \
        dataset="${dataset_config}" \
        data.scene_idx="${scene_id_s_e}" \
        data.data_root="${processed_data_root}/training" \
        data.start_timestep="${start_timestep}" \
        data.end_timestep="${end_timestep}" \
        view_parallel.enabled=true \
        view_parallel.num_view_groups="${num_view_groups}" \
        view_parallel.group_id="${group_id}" \
        > "${log_file}" 2>&1 &

    printf -v "${pid_var}" "%s" "$!"
}

start_eval_group() {
    local group_id="$1"
    local physical_gpu="$2"
    local view_ids="$3"
    local checkpoint="$4"
    local log_file="$5"
    local pid_var="$6"

    setsid env CUDA_VISIBLE_DEVICES="${physical_gpu}" python tools/eval.py \
        --resume_from "${checkpoint}" \
        --view_group_id "${group_id}" \
        --view_ids "${view_ids}" \
        --gpu_id 0 \
        --skip_video_assembly \
        view_parallel.enabled=true \
        view_parallel.num_view_groups="${num_view_groups}" \
        view_parallel.group_id="${group_id}" \
        data.pixel_source.downscale_when_loading="${eval_downscale_when_loading}" \
        > "${log_file}" 2>&1 &

    printf -v "${pid_var}" "%s" "$!"
}

wait_single_job() {
    local stage="$1"
    local pid_var="$2"
    local log_file="${3:-}"

    local pid="${!pid_var}"
    local status=0

    if [ -z "${pid}" ]; then
        log "[ERROR] ${stage}: empty PID"
        exit 1
    fi

    wait "${pid}" || status=$?

    # 进程已经结束，及时清空，避免退出时误杀复用后的 PID。
    printf -v "${pid_var}" "%s" ""

    if [ "${status}" -ne 0 ]; then
        log "[ERROR] ${stage} failed with status ${status}"
        show_log_tail "${log_file}"
        exit 1
    fi

    log "${stage} completed."
}

wait_parallel_jobs() {
    local stage="$1"
    local log0="${2:-}"
    local log1="${3:-}"

    local current_pid0="${pid0}"
    local current_pid1="${pid1}"
    local status0=0
    local status1=0

    wait "${current_pid0}" || status0=$?
    pid0=""

    wait "${current_pid1}" || status1=$?
    pid1=""

    if [ "${status0}" -ne 0 ] || [ "${status1}" -ne 0 ]; then
        log "[ERROR] ${stage} failed:"
        log "  group0 status=${status0}"
        log "  group1 status=${status1}"

        show_log_tail "${log0}"
        show_log_tail "${log1}"

        exit 1
    fi

    log "${stage} completed."
}

run_training_stage() {
    if [[ "${selected_group}" == "group0" ]]; then
        log "Starting Training group 0 only on physical GPU ${group0_gpu}"
        start_train_group \
            0 \
            "${group0_gpu}" \
            "${train_group0_views}" \
            "${group0_train_log}" \
            pid0
        wait_single_job "Training group 0" pid0 "${group0_train_log}"
        return
    fi

    if [ "${group_execution_mode}" = "serial" ]; then
        log "Starting serial view-group training..."

        log "Starting training group 0 on physical GPU ${group0_gpu}"
        start_train_group \
            0 \
            "${group0_gpu}" \
            "${train_group0_views}" \
            "${group0_train_log}" \
            pid0

        log "Started group 0 PID ${pid0}"
        wait_single_job \
            "Training group 0" \
            pid0 \
            "${group0_train_log}"

        log "Starting training group 1 on physical GPU ${group1_gpu}"
        start_train_group \
            1 \
            "${group1_gpu}" \
            "${train_group1_views}" \
            "${group1_train_log}" \
            pid1

        log "Started group 1 PID ${pid1}"
        wait_single_job \
            "Training group 1" \
            pid1 \
            "${group1_train_log}"

        return
    fi

    log "Starting parallel view-group training..."

    if [ "${group0_gpu}" = "${group1_gpu}" ]; then
        log "[WARN] Both groups are using physical GPU ${group0_gpu}."
        log "[WARN] Parallel mode on one GPU may cause CUDA out-of-memory."
    fi

    start_train_group \
        0 \
        "${group0_gpu}" \
        "${train_group0_views}" \
        "${group0_train_log}" \
        pid0

    start_train_group \
        1 \
        "${group1_gpu}" \
        "${train_group1_views}" \
        "${group1_train_log}" \
        pid1

    log "Started group 0 PID ${pid0} on physical GPU ${group0_gpu}"
    log "Started group 1 PID ${pid1} on physical GPU ${group1_gpu}"

    wait_parallel_jobs \
        "Parallel training" \
        "${group0_train_log}" \
        "${group1_train_log}"
}

run_evaluation_stage() {
    if [[ "${selected_group}" == "group0" ]]; then
        log "Starting Evaluation group 0 only on physical GPU ${group0_gpu}"
        start_eval_group \
            0 \
            "${group0_gpu}" \
            "${eval_group0_views}" \
            "${group0_ckpt}" \
            "${group0_eval_log}" \
            pid0
        wait_single_job "Evaluation group 0" pid0 "${group0_eval_log}"
        return
    fi

    if [ "${group_execution_mode}" = "serial" ]; then
        log "Starting serial view-group evaluation..."

        log "Starting evaluation group 0 on physical GPU ${group0_gpu}"
        start_eval_group \
            0 \
            "${group0_gpu}" \
            "${eval_group0_views}" \
            "${group0_ckpt}" \
            "${group0_eval_log}" \
            pid0

        log "Started eval group 0 PID ${pid0}"
        wait_single_job \
            "Evaluation group 0" \
            pid0 \
            "${group0_eval_log}"

        log "Starting evaluation group 1 on physical GPU ${group1_gpu}"
        start_eval_group \
            1 \
            "${group1_gpu}" \
            "${eval_group1_views}" \
            "${group1_ckpt}" \
            "${group1_eval_log}" \
            pid1

        log "Started eval group 1 PID ${pid1}"
        wait_single_job \
            "Evaluation group 1" \
            pid1 \
            "${group1_eval_log}"

        return
    fi

    log "Starting parallel view-group evaluation..."

    if [ "${group0_gpu}" = "${group1_gpu}" ]; then
        log "[WARN] Both evaluation groups are using physical GPU ${group0_gpu}."
        log "[WARN] Parallel evaluation on one GPU may cause CUDA out-of-memory."
    fi

    start_eval_group \
        0 \
        "${group0_gpu}" \
        "${eval_group0_views}" \
        "${group0_ckpt}" \
        "${group0_eval_log}" \
        pid0

    start_eval_group \
        1 \
        "${group1_gpu}" \
        "${eval_group1_views}" \
        "${group1_ckpt}" \
        "${group1_eval_log}" \
        pid1

    log "Started eval group 0 PID ${pid0} on physical GPU ${group0_gpu}"
    log "Started eval group 1 PID ${pid1} on physical GPU ${group1_gpu}"

    wait_parallel_jobs \
        "Parallel evaluation" \
        "${group0_eval_log}" \
        "${group1_eval_log}"
}

trap 'cleanup_children INT; exit 130' INT
trap 'cleanup_children TERM; exit 143' TERM
trap 'cleanup_children EXIT' EXIT

# source /opt/conda/etc/profile.d/conda.sh
# conda activate main

export PYTHONPATH="${repo_root}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${repo_root}/.torch_extensions}"

# Check CUDA before touching an existing run or invoking gsplat's JIT compiler.
check_cuda_device() {
    CUDA_VISIBLE_DEVICES="$1" python - "$1" <<'PY' >&2
import os
import sys
import torch

try:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        raise RuntimeError("PyTorch detects no usable CUDA device")
    torch.empty(1, device="cuda:0")
    print("[INFO] CUDA device {}: {}".format(sys.argv[1], torch.cuda.get_device_name(0)))
except Exception as error:
    print("[ERROR] CUDA_VISIBLE_DEVICES={}: {}".format(os.environ.get("CUDA_VISIBLE_DEVICES"), error))
    print("[ERROR] Check GPU allocation/driver and nvidia-smi; override with GROUP0_GPU=<available GPU ID>.")
    sys.exit(1)
PY
}

check_cuda_device "${group0_gpu}"
if [[ "${selected_group}" == "all" && "${group1_gpu}" != "${group0_gpu}" ]]; then
    check_cuda_device "${group1_gpu}"
fi

mkdir -p "${TORCH_EXTENSIONS_DIR}"

if [ -d "${run_dir}" ]; then
    log "错误：目录 ${run_dir} 已存在，准备删除。"

    if ! rm -rf "${run_dir}"; then
        log "删除 ${run_dir} 失败。可能还有旧训练进程占用 NFS 文件。"
        log "你可以检查："
        log "  lsof +D ${run_dir}"
        log "或者："
        log "  fuser -vm ${run_dir}"
        exit 1
    fi
fi

mkdir -p "${run_dir}"

log "Run directory: ${run_dir}"
log "Selected groups: ${selected_group}"
log "Group execution mode: ${group_execution_mode}"
if [[ "${selected_group}" == "all" ]]; then
    log "Training group 1: physical GPU ${group1_gpu}, views ${train_group1_views}"
fi
log "Training group 0: physical GPU ${group0_gpu}, views ${train_group0_views}"
if [[ "${selected_group}" == "all" ]]; then
    log "Evaluation group 1 views: ${eval_group1_views}"
fi
log "Evaluation group 0 views: ${eval_group0_views}"

log "Preparing gsplat CUDA extension cache: ${TORCH_EXTENSIONS_DIR}"

# 只预编译一次。串行模式下两个训练进程可以复用同一份磁盘缓存。
CUDA_VISIBLE_DEVICES="${group0_gpu}" python -c \
    "from gsplat.cuda._backend import _C; assert _C is not None" \
    >&2

group1_train_log="${run_dir}/group1_train.log"
group0_train_log="${run_dir}/group0_train.log"

if [[ "${selected_group}" == "all" ]]; then
    log "Group 1 train log: ${group1_train_log}"
fi
log "Group 0 train log: ${group0_train_log}"

################################################################################
# Training
################################################################################

run_training_stage

group1_ckpt="${run_dir}/group1/checkpoint_final.pth"
group0_ckpt="${run_dir}/group0/checkpoint_final.pth"

checkpoints=("${group0_ckpt}")
if [[ "${selected_group}" == "all" ]]; then
    checkpoints+=("${group1_ckpt}")
fi
for ckpt in "${checkpoints[@]}"; do
    if [ ! -f "${ckpt}" ]; then
        log "[ERROR] Missing checkpoint: ${ckpt}"
        exit 1
    fi
done

################################################################################
# Evaluation
################################################################################

group1_eval_log="${run_dir}/group1_eval.log"
group0_eval_log="${run_dir}/group0_eval.log"

if [[ "${selected_group}" == "all" ]]; then
    log "Group 1 eval log: ${group1_eval_log}"
fi
log "Group 0 eval log: ${group0_eval_log}"

run_evaluation_stage

################################################################################
# Merge render results
################################################################################

if [[ "${selected_group}" == "group0" ]]; then
    # The QCraft layout leaves missing group1 camera slots black.
    view_order="${eval_group0_views}"
fi
log "Merging render outputs into ${run_dir}/videos"

CUDA_VISIBLE_DEVICES="${group0_gpu}" python tools/merge_view_parallel_render.py \
    --group1_dir "${run_dir}/group1/videos_eval" \
    --group0_dir "${run_dir}/group0/videos_eval" \
    --output_dir "${run_dir}/videos" \
    --view_order "${view_order}" \
    --dataset qcraft \
    --fps "${fps}" \
    --strict \
    >&2

################################################################################
# Merge metrics
################################################################################

log "Merging metrics into ${run_dir}/metrics_eval"

if [[ "${selected_group}" == "group0" ]]; then
    cp -a "${run_dir}/group0/metrics_eval" "${run_dir}/metrics_eval"
else
python tools/merge_view_parallel_metrics.py \
    --group1_metrics_dir "${run_dir}/group1/metrics_eval" \
    --group0_metrics_dir "${run_dir}/group0/metrics_eval" \
    --output_dir "${run_dir}/metrics_eval" \
    --group1_views "${eval_group1_views}" \
    --group0_views "${eval_group0_views}" \
    >&2

fi

trap - INT TERM EXIT

log "Done."
if [[ "${selected_group}" == "all" ]]; then
    log "Group 1 checkpoint: ${group1_ckpt}"
fi
log "Group 0 checkpoint: ${group0_ckpt}"
log "Unified videos: ${run_dir}/videos"
log "Unified metrics: ${run_dir}/metrics_eval"

# stdout 只输出 run_dir，方便 pipeline.sh 捕获。
printf "%s\n" "${run_dir}"
