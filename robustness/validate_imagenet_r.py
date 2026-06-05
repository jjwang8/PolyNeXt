#!/usr/bin/env python3
"""ImageNet-R evaluation (renditions: art, sculptures, etc.).

ImageNet-R covers a 200-class subset of ImageNet-1k, so the 1000-way logits are
restricted to those 200 classes (via a boolean mask) before scoring.

Usage:
    python validate_imagenet_r.py --data /path/to/imagenet-r \
        --checkpoint /path/to/cpolynext_s.pt --config CPolyNeXt_S \
        --init_channels 72 --layers 17
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import InterpolationMode

from polynext_eval import add_model_args, build_and_load, eval_transform
from imagenet_subsets import imagenet_r_mask


def main():
    ap = argparse.ArgumentParser("ImageNet-R evaluation")
    ap.add_argument('--data', type=str, required=True,
                    help='ImageNet-R root (200 wnid class folders)')
    add_model_args(ap)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    tf = eval_transform(232, 224, InterpolationMode.BILINEAR)
    ds = datasets.ImageFolder(args.data, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_and_load(args, device)

    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)[:, imagenet_r_mask]   # restrict to the 200 classes
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

    print(f"\nImageNet-R Top-1 = {100.0 * correct / total:.2f}%  ({correct}/{total})")


if __name__ == '__main__':
    main()
