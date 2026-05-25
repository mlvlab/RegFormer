#!/usr/bin/env bash
set -euo pipefail

# detr r50 : detr-r50-e632da11.pth
# detr r101 : detr-r101-2c7b67e5.pth

cd "$(dirname "$0")/../.."

detector_ckpt=${1:-detr-r50-e632da11.pth}

python main_det_roi.py \
    --world-size 1 \
    --output-dir output/det_extract/hico_r50 \
    --eval \
    --clip_dir_vit checkpoints/pretrained_clip/ViT-B-16.pt \
    --pretrained "params/$detector_ckpt"
