# PolyNeXt - Out-of-Distribution Robustness

Evaluation of the ImageNet-1k pretrained PolyNeXt models on four
out-of-distribution benchmarks, without fine-tuning:

- **ImageNet-C** - common corruptions, reported as AlexNet-normalized mCE (lower is better)
- **ImageNet-A** - natural adversarial examples (200-class subset)
- **ImageNet-R** - renditions: art, sculptures, etc. (200-class subset)
- **ImageNet-Sketch** - sketch representations (full 1000 classes)

These reuse the parent repo's model builder and checkpoint loader, so they take
the same `--config / --init_channels / --layers / --checkpoint` arguments as the
main `validate.py`. Run them from this folder; a GPU is required.

## Expected results (paper)

| Model | Clean | IN-C (mCE) | IN-A | IN-R | IN-Sketch |
|-------|------:|-----------:|-----:|-----:|----------:|
| CPolyNeXt-S | 83.9 | 47.9 | 35.1 | 49.4 | 37.8 |
| APolyNeXt-S | 84.3 | 45.0 | 39.6 | 49.7 | 37.5 |
| CPolyNeXt-B | 84.7 | 44.5 | 42.8 | 52.0 | 40.0 |
| APolyNeXt-B | 84.9 | 42.7 | 46.8 | 52.8 | 41.1 |
| CPolyNeXt-L | 84.9 | 42.5 | 48.3 | 54.5 | 41.8 |
| APolyNeXt-L | 85.2 | 42.9 | 49.2 | 54.0 | 41.8 |

Clean is the standard ImageNet top-1 (from the main results table). Per-model
architecture arguments: S `--init_channels 72 --layers 17`, B `--init_channels
84 --layers 21`, L `--init_channels 96 --layers 24`.

## Datasets

- ImageNet-A: https://github.com/hendrycks/natural-adv-examples
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- ImageNet-C: https://github.com/hendrycks/robustness
- ImageNet-Sketch: https://github.com/HaohanWang/ImageNet-Sketch

ImageNet-A/R are class-folder datasets (200 wnid folders each). ImageNet-C is
laid out as `<root>/<corruption>/<severity>/<wnid>/<images>`.

## Running

ImageNet-A and ImageNet-R (example with CPolyNeXt-S):

```bash
python validate_imagenet_a.py --data path/to/imagenet-a \
    --checkpoint path/to/cpolynext_s.pt --config CPolyNeXt_S --init_channels 72 --layers 17

python validate_imagenet_r.py --data path/to/imagenet-r \
    --checkpoint path/to/cpolynext_s.pt --config CPolyNeXt_S --init_channels 72 --layers 17
```

ImageNet-C (mCE uses the bundled `alexnet_imagenetc_errors.json` by default):

```bash
python validate_imagenet_c.py --imagenetc path/to/imagenet-c \
    --checkpoint path/to/cpolynext_s.pt --config CPolyNeXt_S --init_channels 72 --layers 17
```

ImageNet-Sketch is full 1000-class, so it uses the main `validate.py` from the
repo root (point `--val-dir` at the sketch folder):

```bash
python validate.py --data path/to/imagenet-sketch --val-dir path/to/imagenet-sketch \
    --checkpoint path/to/cpolynext_s.pt --config CPolyNeXt_S --init_channels 72 --layers 17
```

## Notes

- The 200-class subset mappings (ImageNet-A indices and the ImageNet-R wnid
  mask) live in `imagenet_subsets.py`.
- mCE for ImageNet-C is the mean over corruptions of (model error / AlexNet
  error); the AlexNet denominators are the standard values in
  `alexnet_imagenetc_errors.json`.
