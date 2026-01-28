import torch
import torch.nn as nn

logged = False
def poly_init(x:torch.Tensor, batch:bool = False):
  global logged
  if not logged:
    print(f"nn.init.kaiming_normal_(x, nonlinearity=relu)")
    logged = True
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
            **kwargs
    ):
        super().__init__()

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
            attn = F.normalize(attn, p=1, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, H, W, self.attention_dim).permute(0, 3, 1, 2)
        x = self.final_conv(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x