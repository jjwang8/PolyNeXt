"""PolyNeXt backbone for MMSegmentation.

This file inlines a self-contained snapshot of the PolyNeXt classification
model (operations, cells, configs) so the segmentation folder runs standalone
without importing the parent repository. The reference implementation lives in
../../../../model.py and ../../../../operations.py; this snapshot is the
LayerNorm variant used for the ADE20K experiments.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from collections import OrderedDict


# ============================================================
# Gradient norm clipping as a graph operation (compile-safe)
# ============================================================

class GradClipNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_norm):
        ctx.max_norm = max_norm
        return x

    @staticmethod
    def backward(ctx, grad):
        norm = grad.norm()
        clipped_norm = torch.where(norm > ctx.max_norm, ctx.max_norm / (norm + 1e-6), torch.ones_like(norm))
        return grad * clipped_norm, None


def grad_clip_norm(x, max_norm):
    return GradClipNorm.apply(x, max_norm)


# ============================================================
# Operations (inlined from operations.py to keep self-contained)
# ============================================================

def poly_init(x: torch.Tensor, batch: bool = False):
    if batch:
        return
    nn.init.kaiming_normal_(x, nonlinearity="relu")

POLY_INIT_FUNC = poly_init


class ScalePerChannel(nn.Module):
    def __init__(self, num_channels, init_num: float = 1, bias: bool = False):
        super(ScalePerChannel, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1) * init_num)
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1)) if bias else None

    def forward(self, x):
        if self.bias is not None:
            return x * self.scale + self.bias
        return x * self.scale


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, affine: bool = True, bias: bool = True):
        super().__init__(num_channels, elementwise_affine=affine, bias=bias)
        self.bias_tf = bias

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = ((x * x).mean(dim=1, keepdim=True) - (u * u)).clamp(0)
        x = (x - u) * torch.rsqrt(s + self.eps)
        if self.bias_tf:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        else:
            x = x * self.weight.view(1, -1, 1, 1)
        return x


class BufferedDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.register_buffer("p", torch.tensor(float(p), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        keep = (1.0 - self.p).clamp(1e-6, 1.0)
        mask = (torch.rand_like(x, dtype=torch.float32) < keep).to(x.dtype)
        return x * mask / keep


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class Attention(nn.Module):
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()
        self.head_dim = head_dim
        self.poly = True
        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Conv2d(dim, self.attention_dim * 2, 1, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.attention_dim, dim, 1, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = nn.Parameter(
            torch.tensor([-(head_dim ** -0.5) / ((head_dim ** -0.5) - 1)] * self.num_heads).log().view(1, -1, 1, 1))
        self.q_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=5,
                                stride=1, padding=2, groups=self.attention_dim, bias=False)
        self.k_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=5,
                                stride=1, padding=2, groups=self.attention_dim, bias=False)
        self.v_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3,
                                stride=1, padding=1, groups=self.attention_dim, bias=False)
        self.final_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3,
                                    stride=1, padding=1, groups=self.attention_dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, -1, H, W)
        qk, v = qkv.split(self.attention_dim, 1)
        q = self.q_conv(qk).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        k = self.k_conv(qk).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        v = self.v_conv(v).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        if not self.poly:
            x = F.scaled_dot_product_attention(q, k, v,
                                               dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            attn = ((q @ k.transpose(-2, -1)) * self.scale.sigmoid() + 1) ** 4
            attn = F.normalize(attn, p=1, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, H, W, self.attention_dim).permute(0, 3, 1, 2)
        x = self.final_conv(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ============================================================
# Utility: stochastic depth schedule
# ============================================================

def drop_paths(drop_path, layers):
    """Generate per-layer drop path rates."""
    return [x.item() for x in torch.linspace(0, drop_path, layers)]


# ============================================================
# Cells
# ============================================================

class Poly_Cell_Imagenet(nn.Module):
    def __init__(self, config, C: int, nodes: int = 6, stage: int = -1,
                 size: int = 56, drop_path: float = 0.0):
        super(Poly_Cell_Imagenet, self).__init__()
        self.nodes = nodes
        self.C = C
        expansion_conv = config["expansion_conv"][stage]
        expansion_mlp = config["expansion_mlp"][stage]

        self.preprocess0 = ScalePerChannel(C)
        self.preprocess1 = ScalePerChannel(C)
        self.postprocess = LayerNorm2d(C, bias=False)

        self.ops = nn.ModuleList()
        for index in range(self.nodes // 2):
            C_inner = int(C * expansion_conv)
            if stage == 0:
                temp = [nn.Conv2d(C, C_inner, kernel_size=1, padding=0, bias=False),
                        nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=2,
                                  dilation=2, groups=C_inner, bias=False),
                        nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=1,
                                  groups=C_inner, bias=False)]
            else:
                temp = [nn.Conv2d(C, C_inner, kernel_size=1, padding=0, bias=False),
                        nn.Conv2d(C_inner, C_inner, kernel_size=5, stride=1, padding=4,
                                  dilation=2, groups=C_inner, bias=False),
                        nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=1,
                                  groups=C_inner, bias=False)]
            final = nn.Sequential(
                nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=1,
                          groups=C_inner, bias=False),
                nn.Conv2d(C_inner, C, kernel_size=1, padding=0, bias=False))
            POLY_INIT_FUNC(final[0].weight)
            POLY_INIT_FUNC(final[1].weight)
            if config["node_norm"] == "layernorm":
                final.extend([LayerNorm2d(C, bias=False), DropPath(drop_path)])
            for i in temp[:3]:
                POLY_INIT_FUNC(i.weight)
            temp.append(final)

            self.C_inner2 = int(C * expansion_mlp)
            temp.extend([
                nn.Conv2d(C, self.C_inner2, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(self.C_inner2 // 2, C, kernel_size=1, padding=0, bias=False)
            ])
            POLY_INIT_FUNC(temp[-1].weight)
            POLY_INIT_FUNC(temp[-2].weight)
            temp[-1] = nn.Sequential(LayerNorm2d(self.C_inner2 // 2, bias=False),
                                     temp[-1], DropPath(drop_path))
            self.ops.extend(temp)

        scale_vals = torch.tensor(config["sigmoid_scale"])
        self.skip_weight = torch.nn.Parameter(
            (-scale_vals / (scale_vals - 1)).log() if config.get("paired_scale", False) else scale_vals)
        self.scale_type = config.get("paired_scale", False)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        ratios = torch.sigmoid(self.skip_weight)
        polys = self.postprocess(s0 + s1)
        for i in range(self.nodes // 2):
            base = i * 6
            scale_idx = (i, i + 1) if self.scale_type else (i * 2, i * 2 + 1)
            b = self.ops[base](polys)
            polys = self.ops[base + 3](
                self.ops[base + 1](b) * self.ops[base + 2](b).flip(dims=[1])
            ) * ratios[scale_idx[0]] + polys
            high, low = self.ops[base + 4](polys).split(self.C_inner2 // 2, 1)
            polys = self.ops[base + 5](high * low) * ratios[scale_idx[1]] + polys
        return polys


class Atten_Cell_Imagenet(nn.Module):
    def __init__(self, config, C: int, nodes: int = 6, stage: int = -1,
                 size: int = 56, drop_path: float = 0.0):
        super(Atten_Cell_Imagenet, self).__init__()
        self.nodes = nodes
        self.C = C
        expansion_mlp = config["expansion_mlp"][stage]

        self.preprocess0 = ScalePerChannel(C)
        self.preprocess1 = ScalePerChannel(C)
        self.postprocess = LayerNorm2d(C, bias=False)

        self.ops = nn.ModuleList()
        for index in range(self.nodes // 2):
            temp = [nn.Sequential(
                LayerNorm2d(C, bias=False),
                Attention(C, head_dim=32, num_heads=math.ceil(C / 64)),
                DropPath(drop_path)
            )]
            self.C_inner2 = int(C * expansion_mlp)
            temp.extend([
                nn.Conv2d(C, self.C_inner2, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(self.C_inner2 // 2, C, kernel_size=1, padding=0, bias=False)
            ])
            POLY_INIT_FUNC(temp[-1].weight)
            POLY_INIT_FUNC(temp[-2].weight)
            temp[-1] = nn.Sequential(LayerNorm2d(self.C_inner2 // 2, bias=False),
                                     temp[-1], DropPath(drop_path))
            self.ops.extend(temp)

        scale_vals = torch.tensor(config["sigmoid_scale"])
        self.skip_weight = torch.nn.Parameter(
            (-scale_vals / (scale_vals - 1)).log() if config.get("paired_scale", False) else scale_vals)
        self.scale_type = config.get("paired_scale", False)

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        ratios = torch.sigmoid(self.skip_weight)
        polys = self.postprocess(s0 + s1)
        for i in range(self.nodes // 2):
            base = i * 3
            scale_idx = (i, i + 1) if self.scale_type else (i * 2, i * 2 + 1)
            polys = self.ops[base](polys) * ratios[scale_idx[0]] + polys
            high, low = self.ops[base + 1](polys).split(self.C_inner2 // 2, 1)
            polys = self.ops[base + 2](high * low) * ratios[scale_idx[1]] + polys
        return polys


# ============================================================
# Model configs
# ============================================================

CPolyNeXt_T = {
    "nodes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "downsizes": [2, 4, 10],
    "downsize_type": ["sep3x3", "sep3x3"],
    "channels": {2: 2, 4: 4, 10: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [0.5] + [0.5 ** i for i in range(1, 6)],
    "paired_scale": True,
}

CPolyNeXt_S = {
    "nodes": [6] * 3 + [8] * 3 + [8] * 8 + [8] * 3,
    "downsizes": [3, 6, 14],
    "downsize_type": ["sep3x3", "full3x3"],
    "channels": {3: 2, 6: 4, 14: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [float(-i / 2) for i in range(8)],
}

CPolyNeXt_B = {
    "nodes": [8] * 3 + [8] * 5 + [8] * 10 + [8] * 3,
    "downsizes": [3, 8, 18],
    "downsize_type": ["sep3x3", "full3x3"],
    "channels": {3: 2, 8: 4, 18: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [float(-i / 2) for i in range(8)],
}

CPolyNeXt_L = {
    "nodes": [8] * 3 + [8] * 6 + [8] * 12 + [8] * 3,
    "downsizes": [3, 9, 21],
    "downsize_type": ["sep3x3", "full3x3"],
    "channels": {3: 2, 9: 4, 21: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [float(-i / 2 - 0.5) for i in range(8)],
}

APolyNeXt_T = {
    "nodes": [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "downsizes": [2, 4, 10],
    "downsize_type": ["sep3x3", "sep3x3"],
    "channels": {2: 2, 4: 4, 10: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [0.5] + [0.5 ** i for i in range(1, 6)],
    "paired_scale": True,
    "attn": True,
}

APolyNeXt_S = {
    "nodes": [6] * 3 + [8] * 3 + [8] * 8 + [8] * 3,
    "downsizes": [3, 6, 14],
    "downsize_type": ["sep3x3", "full3x3"],
    "channels": {3: 2, 6: 4, 14: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [float(-i / 2) for i in range(8)],
    "attn": True,
}

APolyNeXt_B = {
    "nodes": [8] * 3 + [8] * 5 + [8] * 10 + [8] * 3,
    "downsizes": [3, 8, 18],
    "downsize_type": ["sep3x3", "full3x3"],
    "channels": {3: 2, 8: 4, 18: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [float(-i / 2) for i in range(8)],
    "attn": True,
}

APolyNeXt_L = {
    "nodes": [8] * 3 + [8] * 6 + [8] * 12 + [8] * 3,
    "downsizes": [3, 9, 21],
    "downsize_type": ["sep3x3", "full3x3"],
    "channels": {3: 2, 9: 4, 21: 6},
    "node_norm": "layernorm",
    "expansion_conv": [1, 1, 0.75, 0.75],
    "expansion_mlp": [2, 2, 1.75, 1.75],
    "sigmoid_scale": [float(-i / 2 - 0.5) for i in range(8)],
    "attn": True,
}

CONFIG_REGISTRY = {
    "CPolyNeXt_T": CPolyNeXt_T,
    "CPolyNeXt_S": CPolyNeXt_S,
    "CPolyNeXt_B": CPolyNeXt_B,
    "CPolyNeXt_L": CPolyNeXt_L,
    "APolyNeXt_T": APolyNeXt_T,
    "APolyNeXt_S": APolyNeXt_S,
    "APolyNeXt_B": APolyNeXt_B,
    "APolyNeXt_L": APolyNeXt_L,
}

# ============================================================
# Segmentation backbone wrapper
# ============================================================

@BACKBONES.register_module()
class PolyNeXt(nn.Module):
    """
    PolyNeXt backbone for semantic segmentation (mmseg-compatible).

    Wraps the classification-style NetworkPolyImageNet to output multi-scale
    feature maps as required by FPN-based segmentors (UperNet, etc.).

    The model has 4 stages separated by 3 downsample points.
    The stem does 4x downsample, each stage transition does 2x.
    Output strides: {4, 8, 16, 32}.

    Args:
        init_channels (int): Base channel count C.
        config_name (str): One of the registered config names.
        out_indices (tuple[int]): Which stage outputs to return (0-indexed).
        drop_path_rate (float): Stochastic depth rate.
        norm_eval (bool): If True, freeze BN layers during training.
        pretrained (str | None): Path to pretrained classification weights.
        grad_clip_norm (float | None): If set, clip gradient norm at each cell
            boundary during backward. Prevents gradient explosion in deep
            networks with poorly conditioned weights. Set to None to disable.
    """

    def __init__(self,
                 init_channels=48,
                 config_name="CPolyNeXt_S",
                 out_indices=(0, 1, 2, 3),
                 drop_path_rate=0.1,
                 norm_eval=True,
                 pretrained=None,
                 grad_clip_norm=None):
        super(PolyNeXt, self).__init__()
        self.out_indices = out_indices
        self.norm_eval = norm_eval
        self.pretrained = pretrained
        self.grad_clip_norm = grad_clip_norm

        config = CONFIG_REGISTRY[config_name]
        layers = len(config["nodes"])
        self.drop_path_prob = drop_paths(drop_path_rate, layers)
        self.downsizes = config["downsizes"]

        C_prev = C_curr = init_channels
        self.stem = nn.Conv2d(3, C_prev, 7, stride=4, padding=3, bias=False)

        self.cells = nn.ModuleList()
        self.reductions = nn.ModuleList()
        stage = 0
        size = 56  # nominal; not used at runtime for variable-size inputs

        # Track per-stage output channels for FPN
        self._out_channels = [C_prev]

        for i in range(layers):
            C_curr = int(config["channels"][i] * init_channels) if i in config["channels"] else C_curr
            if i in self.downsizes:
                self.reductions.append(self._create_downsize(C_prev, C_curr, config["downsize_type"][0]))
                self.reductions.append(self._create_downsize(C_prev, C_curr, config["downsize_type"][1]))
                stage += 1
                size //= 2
                self._out_channels.append(C_curr)

            if config.get("attn", False) and stage >= 2:
                cell = Atten_Cell_Imagenet(config, C_curr, config["nodes"][i], stage, size,
                                           self.drop_path_prob[i])
            else:
                cell = Poly_Cell_Imagenet(config, C_curr, config["nodes"][i], stage, size,
                                          self.drop_path_prob[i])
            self.cells.append(cell)
            C_prev = C_curr

        assert len(self._out_channels) == 4, (
            f"Expected 4 stages (3 downsamples), got {len(self._out_channels)} "
            f"from downsizes={self.downsizes}")

        # Per-stage normalization for output features
        self.extra_norms = nn.ModuleList()
        for ch in self._out_channels:
            self.extra_norms.append(LayerNorm2d(ch, bias=True))

    @staticmethod
    def _create_downsize(C_prev, C_curr, down_type):
        if down_type == "sep3x3":
            layer = nn.Sequential(
                nn.Conv2d(C_prev, C_prev, kernel_size=3, stride=2, padding=1,
                          groups=C_prev, bias=False),
                nn.Conv2d(C_prev, C_curr, kernel_size=1, padding=0, bias=False))
            POLY_INIT_FUNC(layer[0].weight)
            POLY_INIT_FUNC(layer[1].weight)
        elif down_type == "full3x3":
            layer = nn.Conv2d(C_prev, C_curr, kernel_size=3, stride=2, padding=1, bias=False)
            POLY_INIT_FUNC(layer.weight)
        elif down_type == "full2x2":
            layer = nn.Conv2d(C_prev, C_curr, kernel_size=2, stride=2, bias=False)
            POLY_INIT_FUNC(layer.weight)
        elif down_type == "sep2x2":
            layer = nn.Sequential(
                nn.Conv2d(C_prev, C_prev, kernel_size=2, stride=2, groups=C_prev, bias=False),
                nn.Conv2d(C_prev, C_curr, kernel_size=1, padding=0, bias=False))
            POLY_INIT_FUNC(layer[0].weight)
            POLY_INIT_FUNC(layer[1].weight)
        else:
            raise NotImplementedError(f"No such downsize: {down_type}")
        return layer

    def init_weights(self, pretrained=None):
        """Initialize weights, optionally loading pretrained classification ckpt."""
        pretrained = pretrained or self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                try:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                except:
                    pass

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()

            # Smart loading that handles various checkpoint formats
            ckpt = torch.load(pretrained, map_location='cpu')
            state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            state = ckpt.get("state_dict_ema", ckpt.get("model_ema", ckpt.get("ema", state)))

            # Strip "module." prefix if present
            if any(k.startswith("module.") for k in state.keys()):
                state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state.items())

            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def _clip(self, x):
        if self.grad_clip_norm is not None:
            return grad_clip_norm(x, self.grad_clip_norm)
        return x

    @property
    def out_channels(self):
        """Return the output channel list for FPN."""
        return [self._out_channels[i] for i in self.out_indices]

    def forward(self, x):
        """
        Returns:
            tuple[Tensor]: Feature maps from each stage in out_indices.
        """
        s0 = s1 = self.stem(x)
        r_idx = 0
        stage = 0
        outs = []

        # Determine last cell index in each stage
        stage_ends = []
        for ds in self.downsizes:
            stage_ends.append(ds - 1)
        stage_ends.append(len(self.cells) - 1)

        current_stage_end_idx = 0

        for i, cell in enumerate(self.cells):
            if i in self.downsizes:
                s0 = self.reductions[r_idx](s0)
                s1 = self.reductions[r_idx + 1](s1)
                r_idx += 2
                stage += 1

            s0, s1 = s1, self._clip(cell(s0, s1, self.drop_path_prob))

            if i == stage_ends[current_stage_end_idx]:
                if current_stage_end_idx in self.out_indices:
                    out = self.extra_norms[current_stage_end_idx](s1)
                    outs.append(out)
                current_stage_end_idx += 1

        return tuple(outs)

    def train(self, mode=True):
        """Keep norm layers frozen during training if norm_eval=True."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()