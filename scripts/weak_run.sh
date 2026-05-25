#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash scripts/weak_run.sh <embed_dim> <vision_encoder> <text_encoder> <input_resolution> [zs_type]"
    echo "Example: bash scripts/weak_run.sh 768 facebook/dinov2-with-registers-small openai/clip-vit-base-patch16 518"
    echo "Example: bash scripts/weak_run.sh 768 facebook/dinov2-with-registers-small openai/clip-vit-base-patch16 518 rare_first"
    echo
    echo "Useful env overrides:"
    echo "  USE_WANDB=1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

embed_dim="$1"
vision_encoder="$2"
text_encoder="$3"
input_resolution="$4"
zs_type="${5:-}"
if [[ "$zs_type" == "full" || "$zs_type" == "none" ]]; then
    zs_type=""
fi

nproc_per_node="${NPROC_PER_NODE:-2}"
num_epochs=5
loss_type="focal"
lr="2e-4"
global_batch_size=32
gradient_accumulation_steps=1
patch_scale=20
instance_score_dim=512
text_ft_start_layer=9
text_ft_end_layer=11
text_lora_r=16
attention_pool_heads=12
attention_pool_dim="$embed_dim"
vision_hidden_dim=$((embed_dim * 4))

logit_scale=20.0
logit_bias=-10.0
instance_scale=10.0
instance_bias=-5.0
instance_prior_factor=0.5

vision_name="${vision_encoder##*/}"
text_name="${text_encoder##*/}"
use_wandb="${USE_WANDB:-0}"
wandb_project="${WANDB_PROJECT:-weak-hoi}"
wandb_group="${WANDB_GROUP:-regformer_weak_train}"

output_dir="output/weak_hoi/hico_hoi/"
if [[ "${vision_encoder,,}" == *"__pretrained__"* ]]; then
    output_dir+="_flayer-1"
fi
output_dir+="ve${vision_name}_froz"
output_dir+="_te${text_name}_tftlora_tft_sl${text_ft_start_layer}_tft_el${text_ft_end_layer}_lora_r${text_lora_r}_lora_alpha32_lora_dropout0.0"
output_dir+="_vptmlp_ln_vhd${vision_hidden_dim}_tptidentity_thd256_out_proj-1"
output_dir+="/attn_pool_attnml_decoder_attn_dim${attention_pool_dim}_attn_heads${attention_pool_heads}_prefavg_patch_pooltoken/"
output_dir+="_mld_qobject_mld_qckptobject_sep_so_isd${instance_score_dim}_norm_obj"
output_dir+="_inst_score_filt_aggpatch_query_actsigmoid_post_normsc_scale${instance_scale}_bias${instance_bias}_prior${instance_prior_factor}_aggtsoftmax"
output_dir+="_cont:zero_ps${patch_scale}_faggsoft"
output_dir+="_sl${logit_scale}_sb${logit_bias}"
output_dir+="/gbs${global_batch_size}_ga${gradient_accumulation_steps}_mg0.0_in_res${input_resolution}_ep${num_epochs}_lr${lr}_wd0.0001"
output_dir+="_cosine_warmup1.0_warmup_start_lr1e-06_min_lr_ratio0.01_step_typeiteration"
output_dir+="_loss_${loss_type}_ic_w1.0_loss_redtarget_mean_focal_alpha0.25_focal_gamma2.0"
if [[ -n "$zs_type" ]]; then
    output_dir+="_zs${zs_type}"
fi

wandb_name="${output_dir#output/}"
master_port=$((10000 + RANDOM % 55535))

cmd=(
    torchrun
    "--nproc_per_node=${nproc_per_node}"
    "--master_port=${master_port}"
    main_weak.py
    --use_ddp
    --train_data_path hicodet/instances_train2015.json
    --train_image_root hicodet/hico_20160224_det/images/train2015
    --val_data_path hicodet/instances_test2015.json
    --val_image_root hicodet/hico_20160224_det/images/test2015
    --dataset_name hico
    --target_type hoi
    --normalize_embeddings
    --vision_encoder "$vision_encoder"
    --freeze_vision_encoder
    --text_encoder "$text_encoder"
    --text_ft_method lora
    --text_ft_start_layer "$text_ft_start_layer"
    --text_ft_end_layer "$text_ft_end_layer"
    --text_lora_r "$text_lora_r"
    --text_lora_alpha 32
    --text_lora_dropout 0.0
    --vision_projector_type mlp_ln
    --vision_hidden_dim "$vision_hidden_dim"
    --text_projector_type identity
    --text_hidden_dim 256
    --projection_dim -1
    --use_attention_pooling
    --attention_type ml_decoder
    --attention_pool_dim "$attention_pool_dim"
    --attention_pool_heads "$attention_pool_heads"
    --prefix_type avg_patch
    --pool_type token
    --ml_decoder_query_type object
    --ml_decoder_query_ckpt object
    --use_seperate_so
    --instance_score_dim "$instance_score_dim"
    --normalize_object_embeddings
    --instance_filter_type score
    --instance_agg_type patch_query
    --instance_activation_type sigmoid
    --post_normalize_instance_scores
    --instance_scale "$instance_scale"
    --instance_bias "$instance_bias"
    --instance_prior_factor "$instance_prior_factor"
    --patch_score_agg_type softmax
    --content_feature_type zero
    --patch_scale "$patch_scale"
    --feature_agg_type soft
    --scale_logit "$logit_scale"
    --scale_bias "$logit_bias"
    --global_batch_size "$global_batch_size"
    --gradient_accumulation_steps "$gradient_accumulation_steps"
    --max_grad_norm 0.0
    --eval_batch_size 64
    --input_resolution "$input_resolution"
    --num_epochs "$num_epochs"
    --lr "$lr"
    --weight_decay 0.0001
    --use_cosine_scheduler
    --warmup_epochs 1.0
    --warmup_start_lr 1e-06
    --min_lr_ratio 0.01
    --scheduler_step_type iteration
    --num_workers 4
    --loss_type "$loss_type"
    --interaction_classifcation_loss_weight 1.0
    --loss_reduction target_mean
    --focal_alpha 0.25
    --focal_gamma 2.0
    --output_dir "$output_dir"
)

if [[ "$use_wandb" == "1" ]]; then
    cmd+=(
        --use_wandb
        --wandb_project "$wandb_project"
        --wandb_name "$wandb_name"
        --wandb_group "$wandb_group"
    )
    if [[ -n "${WANDB_ID:-}" ]]; then
        cmd+=(--wandb_id "$WANDB_ID")
    fi
fi

if [[ "${vision_encoder,,}" == *"__pretrained__"* ]]; then
    cmd+=(--layer_idx -1)
fi
if [[ -n "$zs_type" ]]; then
    cmd+=(--zs_type "$zs_type")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
