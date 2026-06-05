# PolyNeXt - ADE20K Semantic Segmentation

Semantic segmentation experiments for PolyNeXt, built on
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) (v0.13, pinned
for reproducibility). The PolyNeXt backbones are evaluated with a UPerNet head
on ADE20K, following the MetaFormer (ConvFormer/CAFormer) evaluation setup.

## Results

UPerNet, 160K iterations, 512x512 crops. MACs measured at 512x2048.

| Model | Params (M) | MACs (G) | mIoU |
|-------|-----------|----------|------|
| ConvFormer-S18 | 54 | 925 | 48.6 |
| CAFormer-S18 | 54 | 1024 | 48.9 |
| **CPolyNeXt-S** | 54 | 942 | **50.6** |
| **APolyNeXt-S** | 55 | 1121 | 49.9 |

## Installation

This folder is a self-contained MMSegmentation 0.13 fork. The experiments were
run with Python 3.8, PyTorch 2.4.1, and mmcv-full 1.7.2. Because this pairs an
old mmseg with a much newer mmcv, a fresh manual install needs a few small
source-level compatibility patches, so the packed environment below is the
recommended path.

### Recommended: packed environment

A ready-to-use conda environment (built with `conda-pack`, all compatibility
patches already applied) is available as
[openmmlab.tar.gz](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/openmmlab.tar.gz):

```bash
wget https://huggingface.co/JJWCactus/PolyNeXt/resolve/main/openmmlab.tar.gz
mkdir -p openmmlab && tar -xzf openmmlab.tar.gz -C openmmlab
source openmmlab/bin/activate
conda-unpack
```

This works on Linux with a CUDA setup similar to the original; on other
platforms use the manual install below.

### Manual install (fallback)

```bash
conda env create -f environment.yml
conda activate openmmlab
pip install -e .                      # install this mmsegmentation 0.13 fork
```

Note that this old mmseg 0.13 / newer mmcv-full 1.7.2 combination may need
minor source-level compatibility patches (e.g. version asserts) on a fresh
install, which is why the packed environment above is recommended.

## Dataset

Download ADE20K from the
[official website](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and
arrange it as below, then point `data_root` in the config at
`ADEChallengeData2016` (or symlink it):

```
ADEChallengeData2016/
    annotations/
        training/
        validation/
    images/
        training/
        validation/
```

## Pretrained backbones

The backbones are initialized from the ImageNet-1K pretrained PolyNeXt
classification checkpoints,
[cpolynext_s.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_s.pt)
and
[apolynext_s.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/apolynext_s.pt).
Set the `pretrained` field of the config (or pass `--options`):

```python
# in configs/PolyNeXt/upernet_polynext_s_512_160k_ade20k.py
backbone=dict(..., pretrained='path/to/cpolynext_s.pt')
```

## Training

```bash
# bash tools/dist_train.sh <config> <num_gpus>
bash tools/dist_train.sh configs/PolyNeXt/upernet_polynext_s_512_160k_ade20k.py 2
bash tools/dist_train.sh configs/PolyNeXt/upernet_apolynext_s_512_160k_ade20k.py 2
```

## Evaluation

Trained UPerNet checkpoints:
[uper_cpolynexts.pth](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/uper_cpolynexts.pth)
(CPolyNeXt-S) and
[uper_apolynexts.pth](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/uper_apolynexts.pth)
(APolyNeXt-S).

```bash
# bash tools/dist_test.sh <config> <checkpoint> <num_gpus> --eval mIoU
bash tools/dist_test.sh configs/PolyNeXt/upernet_polynext_s_512_160k_ade20k.py \
    path/to/uper_cpolynexts.pth 2 --eval mIoU
```

## FLOPs / MACs

```bash
python tools/get_flops.py configs/PolyNeXt/upernet_polynext_s_512_160k_ade20k.py
```
