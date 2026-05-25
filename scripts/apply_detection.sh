#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/apply_detection.sh <weak_output_dir_or_ckpt> [extra main_det.py args...]"
    echo "Example: bash scripts/apply_detection.sh output/weak_hoi/hico_hoi/.../run_dir"
    echo "Example: bash scripts/apply_detection.sh output/weak_hoi/hico_hoi/.../run_dir/final_model.pth"
    echo
    echo "Useful env overrides:"
    echo "  PRIOR_SCALE_FACTORS=\"0.5 1.0 2.0 0.0\""
    echo "  POST_SUM_SCALES=\"0.5 1.0 2.0 0.0\""
    echo "  CLIP_INPUT_RESOLUTION=518"
    echo "  USE_WANDB=1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

weak_model_input="$1"
shift 1

world_size="${WORLD_SIZE:-2}"
if [[ -d "$weak_model_input" ]]; then
    weak_output_dir="${weak_model_input%/}"
    weak_model_ckpt="${weak_output_dir}/final_model.pth"
elif [[ -f "$weak_model_input" ]]; then
    weak_model_ckpt="$weak_model_input"
    weak_output_dir="$(cd "$(dirname "$weak_model_ckpt")" && pwd)"
else
    weak_output_dir="${weak_model_input%/}"
    weak_model_ckpt="${weak_output_dir}/final_model.pth"
fi

detr_pretrained="${DETR_PRETRAINED:-params/detr-r50-e632da11.pth}"
clip_dir_vit="${CLIP_DIR_VIT:-checkpoints/pretrained_clip/ViT-B-16.pt}"
hicodet_pkl_dir="${HICODET_PKL_DIR:-data/hicodet_pkl_files}"
cache_features="${CACHE_FEATURES:-${hicodet_pkl_dir}/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p}"
if [[ ! -f "$cache_features" && -f "hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p" ]]; then
    cache_features="hicodet_pkl_files/union_embeddings_cachemodel_crop_padding_zeros_vitb16.p"
fi

prior_scale_factors="${PRIOR_SCALE_FACTORS:-1.0}"
post_sum_scales="${POST_SUM_SCALES:-1.0}"
instance_score_post_masking_type="${INSTANCE_SCORE_POST_MASKING_TYPE:-post_sum}"
clip_input_resolution="${CLIP_INPUT_RESOLUTION:-518}"
use_wandb="${USE_WANDB:-0}"
wandb_project="${WANDB_PROJECT:-weak-hoi}"
wandb_group="${WANDB_GROUP:-regformer_detection}"
base_detection_dir="${OUTPUT_DETECTION_DIR:-${weak_output_dir}/detection}"
input_type="full"
image_level_pooling_flag=(--image_level_pooling)
if [[ "${USE_UNION_IMAGE:-0}" == "1" ]]; then
    input_type="union"
    image_level_pooling_flag=()
fi

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
}

require_any() {
    local label="$1"
    shift
    for path in "$@"; do
        if [[ -f "$path" ]]; then
            return 0
        fi
    done
    echo "Missing required file for ${label}. Tried:" >&2
    printf '  %s\n' "$@" >&2
    exit 1
}

if [[ "${SKIP_CHECKS:-0}" != "1" ]]; then
    require_file "$weak_model_ckpt"
    # require_file "$detr_pretrained"
    # require_file "$clip_dir_vit"
    # require_file "$cache_features"
    require_any "HICO test detector boxes" \
        "${hicodet_pkl_dir}/hicodet_test_bbox_R50_detr-r50-e632da11.p" \
        "hicodet_pkl_files/hicodet_test_bbox_R50_detr-r50-e632da11.p"
fi

read -r -a prior_values <<< "$prior_scale_factors"
read -r -a post_sum_values <<< "$post_sum_scales"

for prior_scale_factor in "${prior_values[@]}"; do
    for post_sum_scale in "${post_sum_values[@]}"; do
        port="${PORT:-$((10000 + RANDOM % 55535))}"
        output_detection_dir="${base_detection_dir}/instance_score_post_masking_type${instance_score_post_masking_type}/post_sum_scale${post_sum_scale}/prior_scale_factor${prior_scale_factor}"
        run_name="${RUN_NAME:-${weak_output_dir#output/}_detection_prior${prior_scale_factor}_post${post_sum_scale}}"

        cmd=(
            python
            main_det.py
            --world-size "$world_size"
            --port "$port"
            --pretrained "$detr_pretrained"
            --eval
            --use_multi_hot
            --num_shot 8
            --file1 "$cache_features"
            --clip_dir_vit "$clip_dir_vit"
            --logits_type T
            --trainin_free_option union
            --post_process
            --use_open_clip
            --open_clip_model_name ViT-B-16-SigLIP-512
            --clip_input_resolution "$clip_input_resolution"
            --open_clip_pretrained webli
            --post_process_type sigmoid
            --class_wise_ap
            --no_zero_padding
            --use_attention_reweighting
            --attention_reweighting_start_layer 12
            --input_type "$input_type"
            "${image_level_pooling_flag[@]}"
            --attn_pool_mask
            --use_weak_model
            --instance_score_scheme so_region
            --weak_model_ckpt "$weak_model_ckpt"
            --force_no_attention_modulation
            --instance_score_post_masking_type "$instance_score_post_masking_type"
            --post_sum_scale "$post_sum_scale"
            --prior_scale_factor "$prior_scale_factor"
            --output-dir "$output_detection_dir"
        )

        if [[ -n "${CUSTOM_DETECTOR_RESULTS_PATH:-}" ]]; then
            cmd+=(--custom_detector_results_path "$CUSTOM_DETECTOR_RESULTS_PATH")
        fi

        if [[ "$use_wandb" == "1" ]]; then
            cmd+=(
                --wandb
                --project_name "$wandb_project"
                --group_name "$wandb_group"
                --run_name "$run_name"
            )
            if [[ -n "${WANDB_ID:-}" ]]; then
                cmd+=(--wandb_id "$WANDB_ID")
            fi
        fi

        cmd+=("$@")

        printf 'Running:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}"
    done
done
