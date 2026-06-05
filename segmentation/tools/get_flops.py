# Get MACs/FLOPs and parameter count for a segmentation model.
#
# Uses ptflops. The backend is chosen automatically from the backbone:
# 'pytorch' is more accurate for conv models, 'aten' for attention models
# (per the ptflops docs). Override with --backend if needed (e.g. for
# non-PolyNeXt backbones).

import argparse
from mmcv import Config
from mmseg.models import build_segmentor
from ptflops import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the FLOPs/MACs of a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 2048],
        help='input image size')
    parser.add_argument(
        '--backend',
        choices=['auto', 'pytorch', 'aten'],
        default='auto',
        help="ptflops backend; 'auto' picks pytorch for conv backbones and "
             "aten for attention backbones")
    args = parser.parse_args()
    return args


def resolve_backend(choice, cfg):
    """Pick the ptflops backend. For PolyNeXt, APolyNeXt_* (attention) -> aten,
    CPolyNeXt_* (conv) -> pytorch. Falls back to pytorch otherwise."""
    if choice != 'auto':
        return choice
    config_name = cfg.model.get('backbone', {}).get('config_name', '')
    return 'aten' if config_name.startswith('A') else 'pytorch'


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    backend = resolve_backend(args.backend, cfg)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not supported with {}'.format(
                model.__class__.__name__))

    macs, params = get_model_complexity_info(
        model, input_shape,
        as_strings=False,
        backend=backend,
        print_per_layer_stat=False,
        verbose=False)

    split_line = '=' * 30
    print(f'{split_line}')
    print(f'Input shape: {input_shape}')
    print(f'ptflops backend: {backend}')
    print(f'MACs: {macs / 1e9:.2f} G')
    print(f'Params: {params / 1e6:.2f} M')
    print(f'{split_line}')


if __name__ == '__main__':
    main()
