#!/usr/bin/env python3
"""ImageNet-A evaluation (natural adversarial examples).

ImageNet-A covers a 200-class subset of ImageNet-1k, so the 1000-way logits are
restricted to those 200 classes before scoring.

Usage:
    python validate_imagenet_a.py --data /path/to/imagenet-a \
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
from imagenet_subsets import indices_in_1k


def main():
    ap = argparse.ArgumentParser("ImageNet-A evaluation")
    ap.add_argument('--data', type=str, required=True,
                    help='ImageNet-A root (200 wnid class folders)')
    add_model_args(ap)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    tf = eval_transform(232, 224, InterpolationMode.BICUBIC)
    ds = datasets.ImageFolder(args.data, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = build_and_load(args, device)

    correct = total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            logits = model(images)[:, indices_in_1k]   # restrict to the 200 classes
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)

    print(f"\nImageNet-A Top-1 = {100.0 * correct / total:.2f}%  ({correct}/{total})")


if __name__ == '__main__':
    main()
