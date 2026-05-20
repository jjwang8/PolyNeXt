# PolyNeXt

**Activation-Free Backbones for Image Recognition: Polynomial Alternatives within MetaFormer-Style Vision Models**

[Paper (Coming Soon)]() | [Models (Coming Soon)]()

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
commit your `.env` file, it is excluded by `.gitignore`.

---

## Model Zoo

Pre-trained weights will be released soon.

### ImageNet-1K Pretrained Models

| Model | Params (M) | FLOPs (G) | Top-1 (%) | Checkpoint |
|-------|------------|-----------|-----------|------------|
| CPolyNeXt-T | 6.4 | 1.2 | 80.2 | Coming Soon |
| CPolyNeXt-S | 26 | 4.8 | 83.9 | Coming Soon |
| CPolyNeXt-B | 40 | 8.5 | 84.7 | Coming Soon |
| CPolyNeXt-L | 57 | 12.6 | 84.9 | Coming Soon |
| APolyNeXt-T | 6.5 | 1.3 | 80.9 | Coming Soon |
| APolyNeXt-S | 26 | 5.3 | 84.3 | Coming Soon |
| APolyNeXt-B | 41 | 9.3 | 84.9 | Coming Soon |
| APolyNeXt-L | 57 | 13.3 | 85.2 | Coming Soon |

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

### Single GPU Training

```bash
python train.py \
    --data /path/to/imagenet \
    --set imagenet1000 \
    --config CPolyNeXt_T \
    --init_channels 48 \
    --layers 12 \
    --batch_size 128 \
    --accumulate 8 \ 
    --learning_rate 0.004 \
    --lr_min 0.00001 \
    --epochs 300 \
    --weight_decay 0.01 \
    --opt adamW \
    --auto_aug \
    --cutmix \
    --smooth 0.1 \
    --save CPolyNeXt_T
```

### Multi-GPU Training (Recommended)

We provide training scripts for all model variants in the `scripts/` directory.

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
    train.py               # Single-GPU training script
    train_multiGpu.py      # Multi-GPU distributed training script
    validate.py            # Validation script
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