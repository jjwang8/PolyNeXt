import torch
import torch.nn as nn

def poly_init(x:torch.Tensor, batch:bool = False):
  if batch:
    return
  nn.init.kaiming_normal_(x, nonlinearity="relu")

POLY_INIT_FUNC = poly_init

class ScalePerChannel(nn.Module):
    def __init__(self, num_channels, init_num:float = 1, bias:bool = False):
        super(ScalePerChannel, self).__init__()
        # Initialize a learnable scaling parameter for each channel.
        self.scale = nn.Parameter(torch.ones(1, num_channels, 1, 1) * init_num)
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1)) if bias else None
        
    def forward(self, x):
        if self.bias is not None:
          return x * self.scale + self.bias
        return x * self.scale
    
class LayerNorm2d(nn.LayerNorm):
  def __init__(self, num_channels, affine:bool = True, bias:bool = True):
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
  
class ChannelBatchNorm(nn.Module):
    """
    Polynomial-compatible normalization for the fully-polynomial ("bn") variants.

    Normalize over (N, C) per (H, W), with running stats per (H, W).
    Affine is factorized: per-(H,W) AND per-channel.

      y = norm(x) * (gamma_c * gamma_hw) + (beta_c + beta_hw)

    gamma_c, beta_c shape: (C,)
    gamma_hw, beta_hw shape: (1,1,H,W)

    At inference the running statistics are constants, so the whole op reduces
    to additions and multiplications (see paper App. "Polynomial-Compatible
    Normalization"). The per-(H,W) buffers are created lazily on first forward.
    """
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine_hw: bool = True,
        affine_c: bool = True,
        bias: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine_hw = bool(affine_hw)
        self.affine_c = bool(affine_c)
        self.bias_tf = bool(bias)
        self.track_running_stats = bool(track_running_stats)

        # lazy HW params
        self.gamma_hw = None
        self.beta_hw = None

        # channel params
        if self.affine_c:
            self.gamma_c = nn.Parameter(torch.ones(self.num_channels))
            self.beta_c = None
        else:
            self.register_parameter("gamma_c", None)
            self.register_parameter("beta_c", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.empty(0))
            self.register_buffer("running_var", torch.empty(0))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def _check_input(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected (N,C,H,W), got {tuple(x.shape)}")
        if x.size(1) != self.num_channels:
            raise ValueError(f"Expected C={self.num_channels}, got C={x.size(1)}")

    def _maybe_init(self, H, W, device, dtype):
        if self.track_running_stats:
            if (self.running_mean.numel() == 0) or (self.running_mean.shape[-2:] != (H, W)):
                self.running_mean = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
                self.running_var  = torch.ones((1, 1, H, W), device=device, dtype=dtype)
                self.num_batches_tracked = torch.tensor(0, device=device, dtype=torch.long)

        if self.affine_hw:
            need = (self.gamma_hw is None) or (tuple(self.gamma_hw.shape[-2:]) != (H, W))
            if need:
                self.gamma_hw = nn.Parameter(torch.ones((1, 1, H, W), device=device, dtype=dtype))
                self.beta_hw = nn.Parameter(torch.zeros((1, 1, H, W), device=device, dtype=dtype)) if self.bias_tf else None

    def forward(self, x):
        self._check_input(x)
        N, C, H, W = x.shape
        self._maybe_init(H, W, x.device, x.dtype)

        if self.training:
            mean = x.mean(dim=(0, 1), keepdim=True)
            ex2 = (x * x).mean(dim=(0, 1), keepdim=True)
            var = (ex2 - mean * mean).clamp(min=0.0)

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_mean.mul_(1 - m).add_(m * mean)
                    self.running_var.mul_(1 - m).add_(m * var)
        else:
            if self.track_running_stats and self.running_mean is not None and self.running_mean.numel() != 0:
                mean, var = self.running_mean, self.running_var
            else:
                mean = x.mean(dim=(0, 1), keepdim=True)
                ex2 = (x * x).mean(dim=(0, 1), keepdim=True)
                var = (ex2 - mean * mean).clamp(min=0.0)

        y = (x - mean) * torch.rsqrt(var + self.eps)

        # factorized affine
        if self.affine_hw:
            y = y * self.gamma_hw
            if self.bias_tf and (self.beta_hw is not None):
                y = y + self.beta_hw

        if self.affine_c:
            y = y * self.gamma_c.view(1, C, 1, 1)
            if self.bias_tf and (self.beta_c is not None):
                y = y + self.beta_c.view(1, C, 1, 1)

        return y


class RunningRowSumNorm_Attn(nn.Module):
    """
    Polynomial-compatible replacement for the l1 attention normalization in the
    fully-polynomial ("bn") variants.

    For attn of shape (B, H, T, T):
      denom_batch[b,h,t] = sum_k attn[b,h,t,k]
      running_denom[h,t] = EMA_B(denom_batch)

      out = attn / (denom + eps) * gamma[h,t]

    - No mean subtraction
    - No variance/std
    - Running stats are ONLY the denom (row-sum) per (H,T)
    - Affine is multiplicative ONLY, shape (1,H,T,1), matching running_denom
    - No bias
    """
    def __init__(self, eps=1e-6, momentum=0.1, track_running_stats=True, affine=True, use_abs=False):
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.track_running_stats = bool(track_running_stats)
        self.affine = bool(affine)
        self.use_abs = bool(use_abs)

        self.gamma = None  # (1,H,T,1) lazy init

        if self.track_running_stats:
            self.register_buffer("running_denom", torch.empty(0))  # (1,H,T,1)
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_denom", None)
            self.register_buffer("num_batches_tracked", None)

    def _maybe_init(self, attn: torch.Tensor):
        if attn.dim() != 4:
            raise ValueError(f"Expected (B,H,T,T), got {tuple(attn.shape)}")
        B, H, Tq, Tk = attn.shape
        if Tq != Tk:
            raise ValueError(f"Expected square (T,T), got {Tq} vs {Tk}")

        if self.track_running_stats:
            need = (self.running_denom.numel() == 0) or (self.running_denom.shape[1] != H) or (self.running_denom.shape[2] != Tq)
            if need:
                self.running_denom = torch.ones((1, H, Tq, 1), device=attn.device, dtype=attn.dtype)
                self.num_batches_tracked = torch.tensor(0, device=attn.device, dtype=torch.long)

        if self.affine:
            if (self.gamma is None) or (self.gamma.shape[1] != H) or (self.gamma.shape[2] != Tq):
                self.gamma = nn.Parameter(torch.ones((1, H, Tq, 1), device=attn.device, dtype=attn.dtype))

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        self._maybe_init(attn)

        x = attn.abs() if self.use_abs else attn  # poly attn is nonnegative anyway

        if self.training:
            denom_bht1 = x.sum(dim=-1, keepdim=True)           # (B,H,T,1)
            denom = denom_bht1.mean(dim=0, keepdim=True)       # (1,H,T,1)

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_denom.mul_(1 - m).add_(m * denom)
        else:
            if self.track_running_stats and self.running_denom is not None and self.running_denom.numel() != 0:
                denom = self.running_denom
            else:
                denom = x.sum(dim=-1, keepdim=True).mean(dim=0, keepdim=True)

        out = attn / (denom + self.eps)

        if self.affine and (self.gamma is not None):
            out = out * self.gamma

        return out


def make_norm(norm_type, num_channels, bias=False):
    """Norm factory used throughout the models.

    'ln' -> LayerNorm2d      (default; the main / published models)
    'bn' -> ChannelBatchNorm (fully-polynomial, FHE-oriented variants)
    """
    if norm_type == "bn":
        return ChannelBatchNorm(num_channels, bias=bias)
    return LayerNorm2d(num_channels, bias=bias)


class BufferedDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        # p is a tensor buffer, not a Python float guard
        self.register_buffer("p", torch.tensor(float(p), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        # keep prob as tensor; avoid python branching on p
        keep = (1.0 - self.p).clamp(1e-6, 1.0)         # [scalar tensor]
        # use fp32 rand for good RNG, then cast
        mask = (torch.rand_like(x, dtype=torch.float32) < keep).to(x.dtype)
        return x * mask / keep
  
import torch.nn.functional as F
class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm. https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/metaformer.py
    """

    def __init__(
            self,
            dim,
            head_dim=32,
            num_heads=None,
            qkv_bias=False,
            attn_drop=0.,
            proj_drop=0.,
            proj_bias=False,
            norm="ln",
            **kwargs
    ):
        super().__init__()

        self.norm_type = norm
        self.head_dim = head_dim
        # self.scale = nn.Parameter(torch.tensor([head_dim ** -0.5]))
        
        self.poly = True

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim

        # self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.qkv = nn.Conv2d(dim, self.attention_dim * 2, 1, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.attention_dim, dim, 1, 1, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.shift = nn.Parameter(torch.tensor([1.0]*self.num_heads).view(1, -1, 1, 1))
        self.scale = nn.Parameter(torch.tensor([-(head_dim ** -0.5)/((head_dim ** -0.5) - 1)]*self.num_heads).log().view(1, -1, 1, 1))
        self.q_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=5, stride=1, padding=2, groups=self.attention_dim, bias=False)
        self.k_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=5, stride=1, padding=2, groups=self.attention_dim, bias=False)
        self.v_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3, stride=1, padding=1, groups=self.attention_dim, bias=False)
        self.final_conv = nn.Conv2d(self.attention_dim, self.attention_dim, kernel_size=3, stride=1, padding=1, groups=self.attention_dim, bias=False)

        # Fully-polynomial ("bn") variant: replace the l1 attention normalization
        # with a running row-sum estimate, and add a ChannelBatchNorm before the
        # final projection. Attribute names (norm, norm2) must match the released
        # *_bn checkpoints.
        if self.norm_type == "bn":
            self.norm = RunningRowSumNorm_Attn()
            self.norm2 = ChannelBatchNorm(self.attention_dim, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H*W
        qkv = self.qkv(x).reshape(B, -1, H, W)
        qk, v = qkv.split(self.attention_dim, 1)
        q = self.q_conv(qk).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        k = self.k_conv(qk).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)
        v = self.v_conv(v).reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)

        if not self.poly:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            attn = ((q @ k.transpose(-2, -1))*self.scale.sigmoid() + 1) ** 4
            if self.norm_type == "bn":
                attn = self.norm(attn)
            else:
                attn = F.normalize(attn, p=1, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, H, W, self.attention_dim).permute(0, 3, 1, 2)
        x = self.final_conv(x)
        if self.norm_type == "bn":
            x = self.norm2(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x