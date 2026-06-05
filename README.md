# PolyNeXt

**Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models**

[Paper (arXiv)](https://arxiv.org/abs/2605.20839) | [Models (Hugging Face)](https://huggingface.co/JJWCactus/PolyNeXt)

---

## Overview

PolyNeXt introduces **activation-free polynomial alternatives** for the core modules in modern vision backbones. We replace standard nonlinearities (ReLU, GELU, softmax) with Hadamard products, yielding polynomial functions of the input that provide a natural basis for function approximation without requiring carefully designed activation functions.

We introduce three polynomial modules:

- **PolyMLP**: Replaces the activation in feedforward networks with a Hadamard product of two parallel linear projections
- **PolyConv**: Fuses parallel convolutional branches with different receptive fields via elementwise multiplication
- **PolyAttn**: Replaces the exponential in softmax attention with a polynomial kernel

Combined with lightweight stabilization techniques (Sigmoid-Scale and multi-input skip connections) and a depth-over-width design philosophy, our models match or exceed activation-based counterparts across model scales.

### Key Results

| Model | Params (M) | FLOPs (G) | Top-1 (%) |
|-------|------------|-----------|-----------|
| CPolyNeXt-T | 6.4 | 1.2 | 80.2 |
| APolyNeXt-T | 6.5 | 1.3 | 80.9 |
| CPolyNeXt-S | 26 | 4.8 | 83.9 |
| APolyNeXt-S | 26 | 5.3 | 84.3 |
| CPolyNeXt-B | 40 | 8.5 | 84.7 |
| APolyNeXt-B | 41 | 9.3 | 84.9 |
| CPolyNeXt-L | 57 | 12.6 | 84.9 |
| APolyNeXt-L | 57 | 13.3 | 85.2 |

Our largest model, **APolyNeXt-L**, reaches **85.2% top-1 accuracy** on ImageNet-1K, matching CAFormer-M36 at comparable parameter count. At smaller scale, **APolyNeXt-S** attains **84.3% top-1** at 26M parameters, surpassing CAFormer-S18 (83.6%). Compared to prior polynomial networks (MONet, DTTN), our models improve by 2-3 percentage points at substantially lower computational cost.

---

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- CUDA 12.x (recommended)

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate venv
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

### Experiment Tracking (Weights & Biases)

The training scripts log metrics to [Weights & Biases](https://wandb.ai/) by
default. Provide your API key in a `.env` file in the repository root:

```bash
echo "WANDB_API_KEY=your_key_here" > .env
```

To train without W&B logging, pass the `--wandb_off` flag instead. Do not
commit your `.env` file. It is excluded by `.gitignore`.

---

## Model Zoo

Pre-trained ImageNet-1K weights are available on [Hugging Face](https://huggingface.co/JJWCactus/PolyNeXt).

### ImageNet-1K Pretrained Models

| Model | Params (M) | FLOPs (G) | Top-1 (%) | Checkpoint |
|-------|------------|-----------|-----------|------------|
| CPolyNeXt-T | 6.4 | 1.2 | 80.2 | [cpolynext_t.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_t.pt) |
| CPolyNeXt-S | 26 | 4.8 | 83.9 | [cpolynext_s.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_s.pt) |
| CPolyNeXt-B | 40 | 8.5 | 84.7 | [cpolynext_b.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_b.pt) |
| CPolyNeXt-L | 57 | 12.6 | 84.9 | [cpolynext_l.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_l.pt) |
| APolyNeXt-T | 6.5 | 1.3 | 80.9 | [apolynext_t.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/apolynext_t.pt) |
| APolyNeXt-S | 26 | 5.3 | 84.3 | [apolynext_s.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/apolynext_s.pt) |
| APolyNeXt-B | 41 | 9.3 | 84.9 | [apolynext_b.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/apolynext_b.pt) |
| APolyNeXt-L | 57 | 13.3 | 85.2 | [apolynext_l.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/apolynext_l.pt) |

### Downloading Checkpoints

Download a checkpoint programmatically with `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

ckpt = hf_hub_download(repo_id="JJWCactus/PolyNeXt", filename="cpolynext_s.pt")
```

Or directly with `wget`:

```bash
wget https://huggingface.co/JJWCactus/PolyNeXt/resolve/main/cpolynext_s.pt
```

The downloaded `.pt` file can be passed to `validate.py` via the `--checkpoint` argument.

### Fully-Polynomial (BatchNorm) Variants

These variants replace every LayerNorm with a polynomial-compatible `ChannelBatchNorm` and the attention `l1` normalization with a running row-sum estimate, making the entire inference pass additions and multiplications only (a step toward FHE-compatible inference; see the paper). Accuracies are ImageNet-1K top-1.

| Model | Params (M) | FLOPs (G) | Top-1 (%) | Checkpoint |
|-------|------------|-----------|-----------|------------|
| CPolyNeXt-T BN | 6.4 | 1.2 | 78.3 | [cpolynext_t_bn.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_t_bn.pt) |
| APolyNeXt-T BN | 6.5 | 1.3 | 78.0 | [apolynext_t_bn.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/apolynext_t_bn.pt) |
| CPolyNeXt-S BN | 26 | 4.8 | 82.7 | [cpolynext_s_bn.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_s_bn.pt) |

These checkpoints require the `--norm bn` flag so the model is built with the BatchNorm modules. For example:

```bash
python validate.py \
    --data /path/to/imagenet \
    --checkpoint cpolynext_s_bn.pt \
    --config CPolyNeXt_S \
    --norm bn \
    --init_channels 72 \
    --layers 17
```

The `bn`-variant norms create their parameters lazily on the first forward pass, so `validate.py` runs a dummy forward to materialize them before loading the checkpoint. To train these variants from scratch, use the `*_BN.sh` scripts or pass `--norm bn` to `train_multiGpu.py`.

### Low-Resolution Models

`CPolyNeXt-LR` is a compact 3-stage variant (the `LowRes` config, ~5.5M parameters) trained from scratch at native resolution on smaller datasets. Top-1 accuracy (%):

| Dataset | Resolution | Top-1 (%) | Checkpoint |
|---------|------------|-----------|------------|
| CIFAR-10 | 32x32 | 97.1 | [cpolynext_lr_cifar10.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_lr_cifar10.pt) |
| SVHN | 32x32 | 98.1 | [cpolynext_lr_svhn.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_lr_svhn.pt) |
| Tiny-ImageNet | 64x64 | 74.0 | [cpolynext_lr_tiny_imagenet.pt](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/cpolynext_lr_tiny_imagenet.pt) |

Each checkpoint is dataset-specific (trained from scratch). Evaluate with `validate_lowres.py` and train with the `CPolyNeXt_LR_*.sh` scripts (see the Training and Evaluation sections).

### ADE20K Segmentation Models

PolyNeXt backbones with a UPerNet head on ADE20K (160K iterations, 512x512). MACs measured at 512x2048.

| Model | Params (M) | MACs (G) | mIoU | Checkpoint |
|-------|-----------|----------|------|------------|
| CPolyNeXt-S (UPerNet) | 54 | 942 | 50.6 | [uper_cpolynexts.pth](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/uper_cpolynexts.pth) |
| APolyNeXt-S (UPerNet) | 55 | 1121 | 49.9 | [uper_apolynexts.pth](https://huggingface.co/JJWCactus/PolyNeXt/blob/main/uper_apolynexts.pth) |

Segmentation training and evaluation use a separate MMSegmentation-based codebase in the [`segmentation/`](segmentation/) folder - see [`segmentation/README.md`](segmentation/README.md) for setup, configs, and commands.

### Out-of-Distribution Robustness

The pretrained models are also evaluated, without fine-tuning, on ImageNet-C (mCE, lower is better), ImageNet-A, ImageNet-R, and ImageNet-Sketch.

| Model | IN-C (mCE) | IN-A | IN-R | IN-Sketch |
|-------|-----------:|-----:|-----:|----------:|
| CPolyNeXt-S | 47.9 | 35.1 | 49.4 | 37.8 |
| APolyNeXt-S | 45.0 | 39.6 | 49.7 | 37.5 |
| CPolyNeXt-B | 44.5 | 42.8 | 52.0 | 40.0 |
| APolyNeXt-B | 42.7 | 46.8 | 52.8 | 41.1 |
| CPolyNeXt-L | 42.5 | 48.3 | 54.5 | 41.8 |
| APolyNeXt-L | 42.9 | 49.2 | 54.0 | 41.8 |

Evaluation scripts for these benchmarks are in the [`robustness/`](robustness/) folder - see [`robustness/README.md`](robustness/README.md).

---

## Training

### Data Preparation

Download and extract [ImageNet-1K](https://www.image-net.org/) to a directory with the following structure:

```
/path/to/imagenet/
    train/
        n01440764/
            ...
        n01443537/
            ...
        ...
    val/
        n01440764/
            ...
        n01443537/
            ...
        ...
```

### Multi-GPU Training (Recommended)

We provide training scripts for all model variants in the `scripts/` directory. For a single GPU, run the same `train_multiGpu.py` command with `--nproc_per_node=1`.

**CPolyNeXt-T (2 GPUs):**

```bash
torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:29123 train_multiGpu.py \
    --data /path/to/imagenet \
    --set imagenet1000 \
    --config CPolyNeXt_T \
    --init_channels 48 \
    --layers 12 \
    --batch_size 128 \
    --learning_rate 0.004 \
    --lr_min 0.00001 \
    --epochs 300 \
    --weight_decay 0.01 \
    --opt adamW \
    --auto_aug \
    --cutmix \
    --smooth 0.1 \
    --accumulate 4 \
    --workers 32 \
    --report_freq 20 \
    --save CPolyNeXt_T
```

Or use the provided scripts:

```bash
# Edit the script to set your data path first
bash scripts/CPolyNeXt_T.sh
bash scripts/CPolyNeXt_S.sh
bash scripts/APolyNeXt_T.sh
# ... etc

# Fully-polynomial (BatchNorm) variants - same scripts with --norm bn
bash scripts/CPolyNeXt_T_BN.sh
bash scripts/CPolyNeXt_S_BN.sh
bash scripts/APolyNeXt_T_BN.sh
```

### Training Configurations

| Model | Config | Init Channels | Layers | Optimizer | Learning Rate | Stochastic Depth |
|-------|--------|---------------|--------|-----------|---------------|------------------|
| CPolyNeXt-T | `CPolyNeXt_T` | 48 | 12 | AdamW | 0.004 | 0.00 |
| CPolyNeXt-S | `CPolyNeXt_S` | 72 | 17 | AdamW | 0.004 | 0.20 |
| CPolyNeXt-B | `CPolyNeXt_B` | 84 | 21 | AdamW | 0.004 | 0.30 |
| CPolyNeXt-L | `CPolyNeXt_L` | 96 | 24 | AdamW | 0.004 | 0.50 |
| APolyNeXt-T | `APolyNeXt_T` | 48 | 12 | LAMB | 0.002 | 0.03 |
| APolyNeXt-S | `APolyNeXt_S` | 72 | 17 | LAMB | 0.002 | 0.25 |
| APolyNeXt-B | `APolyNeXt_B` | 84 | 21 | LAMB | 0.002 | 0.40 |
| APolyNeXt-L | `APolyNeXt_L` | 96 | 24 | LAMB | 0.002 | 0.55 |

All models are trained for 300 epochs with batch size 1024 (effective, with gradient accumulation), cosine learning rate schedule, label smoothing (0.1), CutMix, MixUp, and RandAugment.

### Low-Resolution Training (CIFAR-10 / SVHN / Tiny-ImageNet)

The compact `CPolyNeXt-LR` model (a 3-stage variant, the `LowRes` config) is trained from scratch at native resolution with `train_lowres.py` - single-GPU, since these datasets train quickly. The dataset is chosen with `--set`; input resolution, number of classes, and the SVHN no-horizontal-flip rule are all handled automatically.

```bash
python train_lowres.py \
    --set cifar10 \
    --data ./data \
    --config LowRes \
    --init_channels 72 \
    --layers 8 \
    --batch_size 96 \
    --learning_rate 0.001 \
    --epochs 300 \
    --weight_decay 0.05 \
    --opt adamW \
    --auto_aug \
    --cutmix \
    --smooth 0.1 \
    --save CPolyNeXt_LR_CIFAR10
```

Or use the provided scripts:

```bash
bash scripts/CPolyNeXt_LR_CIFAR10.sh
bash scripts/CPolyNeXt_LR_SVHN.sh
bash scripts/CPolyNeXt_LR_TinyImageNet.sh
```

CIFAR-10 and SVHN download automatically into `--data`. For Tiny-ImageNet, point `--data` at an extracted `tiny-imagenet-200` directory containing `train/` and `val/` subfolders.

---

## Evaluation

### Validate on ImageNet-1K

```bash
python validate.py \
    --data /path/to/imagenet \
    --checkpoint /path/to/checkpoint.pt \
    --config CPolyNeXt_T \
    --init_channels 48 \
    --layers 12 \
    --batch_size 128 \
    --workers 4 \
    --gpu 0
```

### Example Output

```
>> 1000 classes found
>=> model and checkpoint loaded

Overall Top-1  = 80.20%
Overall Top-5  = 95.12%

Per-class Top-1 accuracy (sorted):
  [123] n01930112 (nematode): 98.00%
  ...
```

### Validate on CIFAR-10 / SVHN / Tiny-ImageNet

Use `validate_lowres.py` for the `CPolyNeXt-LR` checkpoints:

```bash
python validate_lowres.py \
    --set cifar10 \
    --data ./data \
    --checkpoint cpolynext_lr_cifar10.pt \
    --config LowRes \
    --init_channels 72 \
    --layers 8
```

---

## Model Configurations

### Architecture Overview

PolyNeXt follows a four-stage hierarchical design with decreasing spatial resolution and increasing channel width:

| Stage | Resolution | CPolyNeXt Mixer | APolyNeXt Mixer |
|-------|------------|-----------------|-----------------|
| 1 | H/4 x W/4 | PolyConv | PolyConv |
| 2 | H/8 x W/8 | PolyConv | PolyConv |
| 3 | H/16 x W/16 | PolyConv | PolyAttn |
| 4 | H/32 x W/32 | PolyConv | PolyAttn |

## Repository Structure

```
PolyNeXt/
    environment.yml         # Conda environment specification
    requirements.txt        # Pip requirements
    model.py               # Model definitions (NetworkPolyImageNet, NetworkPoly)
    operations.py          # PolyMLP, PolyConv, PolyAttn implementations
    train_multiGpu.py      # ImageNet-1K training script (multi-GPU; use --nproc_per_node=1 for single-GPU)
    train_lowres.py        # CIFAR-10 / SVHN / Tiny-ImageNet training script
    validate.py            # ImageNet-1K validation script
    validate_lowres.py     # CIFAR-10 / SVHN / Tiny-ImageNet validation script
    utils.py               # Utility functions and transforms
    lamb.py                # LAMB optimizer implementation
    rand_augment.py        # RandAugment implementation
    resize.py              # Resize utilities
    scripts/
        CPolyNeXt_T.sh     # Training script for CPolyNeXt-T
        CPolyNeXt_S.sh     # Training script for CPolyNeXt-S
        CPolyNeXt_B.sh     # Training script for CPolyNeXt-B
        CPolyNeXt_L.sh     # Training script for CPolyNeXt-L
        APolyNeXt_T.sh     # Training script for APolyNeXt-T
        APolyNeXt_S.sh     # Training script for APolyNeXt-S
        APolyNeXt_B.sh     # Training script for APolyNeXt-B
        APolyNeXt_L.sh     # Training script for APolyNeXt-L
        CPolyNeXt_T_BN.sh  # Fully-polynomial (BatchNorm) variant
        CPolyNeXt_S_BN.sh  # Fully-polynomial (BatchNorm) variant
        APolyNeXt_T_BN.sh  # Fully-polynomial (BatchNorm) variant
        CPolyNeXt_LR_CIFAR10.sh       # Low-resolution: CIFAR-10
        CPolyNeXt_LR_SVHN.sh          # Low-resolution: SVHN
        CPolyNeXt_LR_TinyImageNet.sh  # Low-resolution: Tiny-ImageNet
    segmentation/          # ADE20K semantic segmentation (MMSegmentation-based)
    robustness/            # ImageNet-C / -A / -R / -Sketch evaluation
```

---

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang2026polynext,
  title     = {Activation-Free Backbones for Image Recognition: Polynomial
               Alternatives within MetaFormer-Style Vision Models},
  author    = {Wang, Jeffrey and Gregory, Jonathan and Chrysos, Grigorios G.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

> The BibTeX entry will be updated with the official PMLR proceedings reference once published.

---

## Acknowledgements

This codebase builds upon the following works:

- [MetaFormer](https://github.com/sail-sg/metaformer) (Yu et al., 2024)
- [MONet](https://github.com/Allencheng97/Multilinear_Operator_Networks) (Cheng et al., 2024)
- [timm](https://github.com/huggingface/pytorch-image-models) (Wightman, 2019)

We thank the authors for their excellent codebases.

---

## Contact

For questions or issues, please open an issue on GitHub.