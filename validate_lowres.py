"""Validation for the low-resolution CPolyNeXt-LR checkpoints.

Evaluates a trained CPolyNeXt-LR model on CIFAR-10, SVHN, or Tiny-ImageNet at
native resolution. For ImageNet-1K validation use validate.py instead.
"""
import os
import argparse
from collections import OrderedDict

import torch
import torchvision.datasets as dset
from torch.utils.data import DataLoader

from model import *
import utils

NUM_CLASSES = {"cifar10": 10, "svhn": 10, "tiny_imagenet": 200}
INPUT_SIZE  = {"cifar10": 32, "svhn": 32, "tiny_imagenet": 64}


def smart_load(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    if any(k.startswith("module.") for k in state.keys()):
        state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model


def main():
    parser = argparse.ArgumentParser("low-resolution validation")
    parser.add_argument('--data',         type=str, default='./data',
                        help='dataset root (CIFAR-10/SVHN download here; Tiny-ImageNet folder)')
    parser.add_argument('--checkpoint',   type=str, required=True, help='path to .pt checkpoint')
    parser.add_argument('--set',          type=str, default='cifar10',
                        choices=['cifar10', 'svhn', 'tiny_imagenet'])
    parser.add_argument('--config',       type=str, default='LowRes')
    parser.add_argument('--init_channels', type=int, default=72)
    parser.add_argument('--layers',       type=int, default=8)
    parser.add_argument('--batch_size',   type=int, default=128)
    parser.add_argument('--workers',      type=int, default=4)
    parser.add_argument('--gpu',          type=int, default=0)
    args = parser.parse_args()

    num_classes = NUM_CLASSES[args.set]
    size = INPUT_SIZE[args.set]
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # validation transform: native resolution, per-dataset normalization only
    _, valid_tf = utils.transforms_lowres(size)
    if args.set == "cifar10":
        val_ds = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_tf)
    elif args.set == "svhn":
        val_ds = dset.SVHN(root=args.data, split="test", download=True, transform=valid_tf)
    else:
        val_ds = dset.ImageFolder(os.path.join(args.data, "val"), transform=valid_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # LowRes is LayerNorm-only, so no lazy-parameter materialization is needed.
    model = NetworkPoly(args.init_channels, num_classes, args.layers, dict(eval(args.config)), 0.0, 0.0)
    smart_load(model, args.checkpoint)
    model.to(device).eval()
    print(">=> model and checkpoint loaded")

    top1 = 0
    total = 0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            top1 += (pred == targets).sum().item()
            total += targets.size(0)

    print(f"\n{args.set}: Top-1 = {top1 / total * 100:.2f}%  ({total} images)")


if __name__ == '__main__':
    main()
