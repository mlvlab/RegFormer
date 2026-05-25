# [CVPR2026] RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised Human-Object Interaction Detection

> Jihwan Park<sup>1</sup>, Chanhyeong Yang<sup>2</sup>, Jinyoung Park<sup>1</sup>, Taehoon Song<sup>1</sup>,  Hyunwoo J. Kim<sup>1</sup>
>
> <sup>1</sup>KAIST    <sup>2</sup>LG Energy Solution

## Installation

Create and activate the conda environment:

```bash
conda create -n regformer python=3.9
conda activate regformer
```

Install PyTorch:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Install [`pocket`](https://github.com/fredzzhang/pocket), which is required by the detection code inherited from UPT:

```bash
git clone https://github.com/fredzzhang/pocket.git ../pocket
pip install -e ../pocket
```

If `pocket` is already available locally, install that checkout instead:

```bash
pip install -e /path/to/pocket
```

## Data

Download HICO-DET by following the data preparation instructions in the [UPT repository](https://github.com/fredzzhang/upt). Place the annotations and images under `hicodet/`:

```text
hicodet/
  instances_train2015.json
  instances_test2015.json
  hico_20160224_det/
    images/
      train2015/
      test2015/
```

Detection file extraction expects the detector checkpoint at:

```text
params/detr-r50-e632da11.pth
```

Download the DETR R50 checkpoint from the official [DETR model zoo](https://github.com/facebookresearch/detr#model-zoo):

```bash
mkdir -p params
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth -O params/detr-r50-e632da11.pth
```

## Training

Train the weak HOI model in full mode with:

```bash
bash scripts/weak_run.sh 768 facebook/dinov2-with-registers-small openai/clip-vit-base-patch16 518
```

For zero-shot settings, pass the training split mode as the fifth argument:

```bash
# RF-UC
bash scripts/weak_run.sh 768 facebook/dinov2-with-registers-small openai/clip-vit-base-patch16 518 rare_first

# NF-UC
bash scripts/weak_run.sh 768 facebook/dinov2-with-registers-small openai/clip-vit-base-patch16 518 non_rare_first
```

Arguments:

```text
<embed_dim>         attention/pooling embedding dimension
<vision_encoder>   Hugging Face vision encoder name
<text_encoder>     Hugging Face text encoder name
<input_resolution> image input resolution
[zs_type]          optional zero-shot split; use rare_first for RF-UC, non_rare_first for NF-UC
```

Checkpoints and logs are written under `output/weak_hoi/`.

## Detection File Extraction

Before running detection evaluation, extract detector boxes for HICO-DET:

```bash
bash scripts/det_extract/hico_r50.sh
```

This uses `params/detr-r50-e632da11.pth` from the official [DETR repository](https://github.com/facebookresearch/detr) by default and writes:

```text
data/hicodet_pkl_files/hicodet_test_bbox_R50_detr-r50-e632da11.p
```


## Detection Evaluation

After training, apply the detector using the weak output directory:

```bash
bash scripts/apply_detection.sh {weak_output_dir}/final_model.pth
```

Detection outputs are saved under `{weak_output_dir}/detection/` by default.

## Acknowledgements

This repository builds on components from [ADA-CM](https://github.com/ltttpku/ADA-CM) and [UPT](https://github.com/fredzzhang/upt). We thank the authors for releasing their code.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{park2026regformer,
  title     = {RegFormer: Transferable Relational Grounding for Efficient Weakly-Supervised Human-Object Interaction Detection},
  author    = {Park, Jihwan and Yang, Chanhyeong and Park, Jinyoung and Song, Taehoon and Kim, Hyunwoo J.},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```
