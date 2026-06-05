#!/usr/bin/env python3
"""ImageNet-C evaluation (common corruptions) with AlexNet-normalized mCE.

Expects the standard ImageNet-C layout:
    <imagenetc>/<corruption>/<severity>/<wnid>/<images>
with severity in 1..5 over the 15 standard corruptions.

mCE is the mean over corruptions of (model top-1 error / AlexNet top-1 error),
where the AlexNet per-corruption errors come from --mce-ref (defaults to the
bundled alexnet_imagenetc_errors.json). Lower is better.

Usage:
    python validate_imagenet_c.py --imagenetc /path/to/imagenet-c \
        --checkpoint /path/to/cpolynext_s.pt --config CPolyNeXt_S \
        --init_channels 72 --layers 17
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import InterpolationMode

from polynext_eval import add_model_args, build_and_load, eval_transform

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

DEFAULT_MCE_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "alexnet_imagenetc_errors.json")


@torch.no_grad()
def top1(model, loader, device):
    correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        pred = model(images).argmax(1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser("ImageNet-C evaluation")
    ap.add_argument('--imagenetc', type=str, required=True,
                    help='ImageNet-C root (contains the corruption folders)')
    ap.add_argument('--mce-ref', type=str, default=DEFAULT_MCE_REF,
                    help='JSON of AlexNet per-corruption top-1 error (%); '
                         'defaults to the bundled file')
    add_model_args(ap)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    with open(args.mce_ref) as f:
        alexnet_err = {k.strip(): float(v) for k, v in json.load(f).items()}

    tf = eval_transform(224, None, InterpolationMode.BICUBIC)
    model = build_and_load(args, device)

    print("\n=== ImageNet-C ===")
    ce_values = []
    for corr in CORRUPTIONS:
        sev_top1 = []
        for sev in range(1, 6):
            root = os.path.join(args.imagenetc, corr, str(sev))
            if not os.path.isdir(root):
                print(f"[skip] missing {corr}/{sev} under {args.imagenetc}")
                continue
            ds = datasets.ImageFolder(root, transform=tf)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
            sev_top1.append(top1(model, loader, device))
        if not sev_top1:
            print(f"[skip] {corr}: no severities found")
            continue
        corr_top1 = sum(sev_top1) / len(sev_top1)
        ce = (100.0 - corr_top1) / alexnet_err[corr]
        ce_values.append(ce)
        print(f"  {corr:18s} top-1 {corr_top1:6.2f}%   CE {100.0 * ce:6.2f}")

    if ce_values:
        mce = 100.0 * sum(ce_values) / len(ce_values)
        print(f"\nmCE (AlexNet-normalized, lower is better) = {mce:.2f}")
    else:
        print("\nNo corruptions evaluated.")


if __name__ == '__main__':
    main()
