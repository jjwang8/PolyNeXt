import os
import argparse
import json
from typing import OrderedDict
import urllib.request
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from model import *
from utils import transforms_imagenet

def build_model(args, num_classes):
    cfg = dict(eval(args.config))
    cfg["norm"] = args.norm
    return NetworkPolyImageNet(args.init_channels, num_classes,
                               args.layers, cfg, args.dropout)

def smart_load(model, ckpt_path, strict=False):
    ckpt = torch.load(ckpt_path)

    # Common checkpoint field names
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = ckpt.get("state_dict_ema", ckpt.get("model_ema", ckpt.get("ema", state)))

    # Strip a single leading "module." if present
    if any(k.startswith("module.") for k in state.keys()):
        new_state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())
    else:
        new_state = state

    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if not strict:
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
    return model

def main():
    parser = argparse.ArgumentParser("ImageNet-1k validation")
    parser.add_argument('--data',        type=str, required=True,
                        help='root dir of ImageNet1k (should contain "val/")')
    parser.add_argument('--val-dir',     type=str, default=None,
                        help='[optional] explicit val folder if different from data/val')
    parser.add_argument('--checkpoint',  type=str, required=True,
                        help='path to .pt or .pth checkpoint')
    parser.add_argument('--batch_size',  type=int, default=128)
    parser.add_argument('--workers',     type=int, default=4)
    parser.add_argument('--gpu',         type=int, default=0)
    parser.add_argument('--init_channels', type=int, default=48)
    parser.add_argument('--layers',      type=int, default=12)
    parser.add_argument('--config',      type=str, default='CPolyNeXt_T')
    parser.add_argument('--norm',        type=str, default='ln', choices=['ln', 'bn'],
                        help="normalization: 'ln' (default models) or 'bn' (fully-polynomial variants)")
    parser.add_argument('--input_size',  type=int, default=224,
                        help='input resolution; used to materialize lazy bn-variant params before loading')
    parser.add_argument('--poly',        action='store_true',
                        help='use poly model for ImageNet')
    parser.add_argument('--auto_aug',    action='store_true',
                        help='(train-time augmentation; unused in validation)')
    parser.add_argument('--rand_interp', action='store_true',
                        help='(train-time augmentation; unused in validation)')
    parser.add_argument('--dropout',     type=float, default=0.0)
    parser.add_argument('--wnid-json',   type=str,
                        default='imagenet_class_index.json',
                        help='local path or will auto-download if missing')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    # 1) prepare data loader
    val_folder = args.val_dir if args.val_dir else os.path.join(args.data, 'val')
    _, valid_tf = transforms_imagenet(args)
    val_ds = datasets.ImageFolder(val_folder, transform=valid_tf)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    num_classes = len(val_ds.classes)
    print(f">> {num_classes} classes found")

    # 2) load human-readable labels
    if not os.path.exists(args.wnid_json):
        print(f"downloading {args.wnid_json} ...")
        url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
        urllib.request.urlretrieve(url, args.wnid_json)
    with open(args.wnid_json) as f:
        class_idx = json.load(f)
    wnid_to_human = {v[0]: v[1] for v in class_idx.values()}

    idx_to_wnid = {i: wnid for i, wnid in enumerate(val_ds.classes)}
    idx_to_label = {i: wnid_to_human.get(idx_to_wnid[i], idx_to_wnid[i])
                    for i in idx_to_wnid}

    # 3) build and load model
    model = build_model(args, num_classes)
    # bn-variant norms create their parameters lazily on first forward, so a
    # dummy pass is required before load_state_dict (no-op for ln models).
    materialize_lazy_params(model, input_size=args.input_size)
    smart_load(model, args.checkpoint)
    model.to(device).eval()
    print(">=> model and checkpoint loaded")

    # 4) run validation
    total_top1 = 0
    total_top5 = 0
    total_samples = 0

    class_correct = defaultdict(int)
    class_total   = defaultdict(int)

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            _, pred5 = logits.topk(5, 1, True, True)
            pred1 = pred5[:, 0]

            total_samples += targets.size(0)
            total_top1 += (pred1 == targets).sum().item()
            total_top5 += sum((pred5[i] == targets[i]).any().item()
                               for i in range(targets.size(0)))

            for t, p in zip(targets, pred1):
                class_total[t.item()]   += 1
                class_correct[t.item()] += (p == t).item()

    # 5) report overall
    top1_acc = total_top1 / total_samples * 100
    top5_acc = total_top5 / total_samples * 100
    print(f"\nOverall Top-1  = {top1_acc:.2f}%")
    print(f"Overall Top-5  = {top5_acc:.2f}%\n")

    # per-class accuracy sorted by descending Top-1
    results = []
    for idx in range(num_classes):
        if class_total[idx] > 0:
            acc = class_correct[idx] / class_total[idx] * 100
            results.append((idx, idx_to_wnid[idx], idx_to_label[idx], acc))
    results.sort(key=lambda x: x[3], reverse=True)

    print("Per-class Top-1 accuracy (sorted):")
    for idx, wnid, label, acc in results:
        print(f"  [{idx:3d}] {wnid} ({label:>15s}): {acc:5.2f}%")

if __name__ == '__main__':
    main()
