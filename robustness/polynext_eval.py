"""Shared helpers for the robustness evaluations.

Reuses the repo's model builder and checkpoint loader (from the parent
directory) so there is a single source of truth for how PolyNeXt models are
constructed and loaded.
"""
import argparse
import os
import sys

# make the parent repo importable (model.py, validate.py) regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from validate import build_model, smart_load
from model import materialize_lazy_params

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def eval_transform(resize=232, crop=224, interpolation=InterpolationMode.BICUBIC):
    """Resize, optional center-crop, then normalize. Pass crop=None to skip the
    center crop."""
    steps = [transforms.Resize(resize, interpolation=interpolation)]
    if crop is not None:
        steps.append(transforms.CenterCrop(crop))
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def add_model_args(parser):
    """Model / loader arguments, mirroring validate.py."""
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to the .pt/.pth classification checkpoint')
    parser.add_argument('--config', type=str, default='CPolyNeXt_S',
                        help='model config name, e.g. CPolyNeXt_S / APolyNeXt_B')
    parser.add_argument('--init_channels', type=int, default=72)
    parser.add_argument('--layers', type=int, default=17)
    parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn'])
    parser.add_argument('--input_size', type=int, default=224,
                        help='used to materialize lazy bn-variant params before loading')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    return parser


def build_and_load(args, device):
    """Build the model from args, materialize lazy params, load the checkpoint."""
    model = build_model(args, 1000)
    materialize_lazy_params(model, input_size=args.input_size)
    smart_load(model, args.checkpoint)
    return model.to(device).eval()
