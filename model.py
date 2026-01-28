import math
import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path, DropPath, drop_paths

APolyNeXt_T = {
  "nodes": [6, 6,    6, 6,   6, 6, 6, 6, 6, 6,   6, 6],
  "downsizes": [2, 4, 10],
  "downsize_type": ["sep3x3", "sep3x3"],
  "channels": {2: 2, 4: 4, 10: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [0.5] + [0.5 ** i for i in range(1, 6)],
  "paired_scale": True,
  "Attn": True,
}

APolyNeXt_S = {
  "nodes": [6]*3 + [8]*3 + [8]*8 + [8]*3,
  "downsizes": [3, 6, 14],
  "downsize_type": ["sep3x3", "full3x3"],
  "channels": {3: 2, 6: 4, 14: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [float(-i/2) for i in range(8)],
  "Attn": True,
}

APolyNeXt_B = {
  "nodes": [8]*3 + [8]*5 + [8]*10 + [8]*3,
  "downsizes": [3, 8, 18],
  "downsize_type": ["sep3x3", "full3x3"],
  "channels": {3: 2, 8: 4, 18: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [float(-i/2) for i in range(8)],
  "Attn": True,
}


APolyNeXt_L = {
  "nodes": [8]*3 + [8]*6 + [8]*12 + [8]*3,
  "downsizes": [3, 9, 21],
  "downsize_type": ["sep3x3", "full3x3"],
  "channels": {3: 2, 9: 4, 21: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [float(-i/2 - 0.5) for i in range(8)],
  "Attn": True,
}

# Convs =============================

CPolyNeXt_T = {
  "nodes": [6, 6,    6, 6,   6, 6, 6, 6, 6, 6,   6, 6],
  "downsizes": [2, 4, 10],
  "downsize_type": ["sep3x3", "sep3x3"],
  "channels": {2: 2, 4: 4, 10: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [0.5] + [0.5 ** i for i in range(1, 6)],
  "paired_scale": True,
}

CPolyNeXt_S = {
  "nodes": [6]*3 + [8]*3 + [8]*8 + [8]*3,
  "downsizes": [3, 6, 14],
  "downsize_type": ["sep3x3", "full3x3"],
  "channels": {3: 2, 6: 4, 14: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [float(-i/2) for i in range(8)],
}

CPolyNeXt_B = {
  "nodes": [8]*3 + [8]*5 + [8]*10 + [8]*3,
  "downsizes": [3, 8, 18],
  "downsize_type": ["sep3x3", "full3x3"],
  "channels": {3: 2, 8: 4, 18: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [float(-i/2) for i in range(8)],
}


CPolyNeXt_L = {
  "nodes": [8]*3 + [8]*6 + [8]*12 + [8]*3,
  "downsizes": [3, 9, 21],
  "downsize_type": ["sep3x3", "full3x3"],
  "channels": {3: 2, 9: 4, 21: 6},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [float(-i/2 - 0.5) for i in range(8)],
}


class Poly_Cell_Imagenet(nn.Module):
  def __init__(self, config, C:int, nodes:int = 6, stage:int = -1, size:int = 56, drop_path:float = 0.0):
    super(Poly_Cell_Imagenet, self).__init__()
    self.nodes = nodes
    self.C = C
    expansion_conv = config["expansion_conv"] if stage <= 1 else 0.75
    expansion_mlp = config["expansion_mlp"] if stage <= 1 else 1.75
    
    print(C)
    print(drop_path)

    self.preprocess0 = ScalePerChannel(C)
    self.preprocess1 = ScalePerChannel(C)
    self.postprocess = LayerNorm2d(C, bias=False)

    self.ops = nn.ModuleList()
    for index in range(self.nodes//2):
      C_inner = int(C*expansion_conv)
      if stage == 0:
        temp = [nn.Conv2d(C, C_inner, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=2, dilation=2, groups=C, bias=False),
                nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=1, groups=C, bias=False)]
      else:
        temp = [nn.Conv2d(C, C_inner, kernel_size=1, padding=0, bias=False),
                nn.Conv2d(C_inner, C_inner, kernel_size=5, stride=1, padding=4, dilation=2, groups=C, bias=False),
                nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=1, groups=C, bias=False)]
      final = nn.Sequential(nn.Conv2d(C_inner, C_inner, kernel_size=3, stride=1, padding=1, groups=C, bias=False),
                            nn.Conv2d(C_inner, C, kernel_size=1, padding=0, bias=False))
      POLY_INIT_FUNC(final[0].weight)
      POLY_INIT_FUNC(final[1].weight)
      if config["node_norm"] == "layernorm":
        final.extend([LayerNorm2d(C, bias=False), DropPath(drop_path)])
      for i in temp[:3]:
        POLY_INIT_FUNC(i.weight)
      temp.append(final)

      self.C_inner2 = int(C*expansion_mlp)
      temp.extend([nn.Conv2d(C, self.C_inner2, kernel_size=1, padding=0, bias=False), nn.Conv2d(self.C_inner2//2, C, kernel_size=1, padding=0, bias=False)])
      POLY_INIT_FUNC(temp[-1].weight)
      POLY_INIT_FUNC(temp[-2].weight)
      temp[-1] = nn.Sequential(LayerNorm2d(self.C_inner2//2, bias=False), temp[-1], DropPath(drop_path))
      self.ops.extend(temp)
    
    scale_vals = torch.tensor(config["sigmoid_scale"])
    self.skip_weight = torch.nn.Parameter((-scale_vals/(scale_vals - 1)).log() if config.get("paired_scale", False) else scale_vals)
    self.scale_type = config.get("paired_scale", False)
  
  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    ratios = torch.sigmoid(self.skip_weight)

    polys = self.postprocess(s0+s1)
    for i in range(self.nodes//2):
      base = i*6
      scale_idx = (i, i+1) if self.scale_type else (i*2, i*2+1)
      b = self.ops[base](polys)
      polys = self.ops[base+3](self.ops[base+1](b)*self.ops[base+2](b).flip(dims=[1]))*ratios[scale_idx[0]] + polys

      high, low = self.ops[base+4](polys).split(self.C_inner2//2, 1)
      polys = self.ops[base+5](high*low)*ratios[scale_idx[1]] + polys
    return polys
    
class NetworkPolyImageNet(nn.Module):
  def __init__(self, C:int, num_classes:int, layers:int, config:dict, dropout:float=0, drop_path:float = 0.0):
    super(NetworkPolyImageNet, self).__init__()
    self.drop_path_prob = drop_paths(drop_path, layers)
    self.downsizes:list = config["downsizes"]

    C_prev = C_curr = C
    self.stem = nn.Conv2d(3, C, 7, stride=4, padding=3, bias=False)

    self.cells = nn.ModuleList()
    self.reductions = nn.ModuleList()
    stage = 0
    size = 56
    for i in range(layers):
      C_curr = int(config["channels"][i]*C) if i in config["channels"] else C_curr
      if i in self.downsizes:
        self.reductions.append(self.create_downsize(C_prev, C_curr, config["downsize_type"][0]))
        self.reductions.append(self.create_downsize(C_prev, C_curr, config["downsize_type"][1]))
        stage += 1
        size //= 2
      if config["attn"] and stage >= 2:
        cell = Atten_Cell_Imagenet(config, C_curr, config["nodes"][i], stage, size, self.drop_path_prob[i])
      else:
        cell = Poly_Cell_Imagenet(config, C_curr, config["nodes"][i], stage, size, self.drop_path_prob[i])
      self.cells.append(cell)
      C_prev = C_curr

    self.norm = LayerNorm2d(C_prev, bias=False)
    self.proj = nn.Conv2d(C_prev, C_prev*4, 1, 1, 0, bias=False)
    self.head = nn.Sequential(  
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      BufferedDropout(p=dropout),
      nn.Linear(C_prev*2, num_classes)
    )

  def create_downsize(self, C_prev, C_curr, down_type):
    if down_type == "sep3x3":
      layer = nn.Sequential(nn.Conv2d(C_prev, C_prev, kernel_size=3, stride=2, padding=1, groups=C_prev, bias=False),
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
      layer = nn.Sequential(nn.Conv2d(C_prev, C_prev, kernel_size=2, stride=2, groups=C_prev, bias=False),
                            nn.Conv2d(C_prev, C_curr, kernel_size=1, padding=0, bias=False))
      POLY_INIT_FUNC(layer[0].weight)
      POLY_INIT_FUNC(layer[1].weight)
    else:
      raise NotImplementedError(f"No such downsize: {down_type}")
    return layer
  
  def set_head_dropout(self, prob: float):
    p_buf = self.head[2].p
    p_buf.copy_(p_buf.new_tensor(float(prob)))

  def forward(self, input):
    s0 = s1 = self.stem(input)
    r_idx = 0
    for i, cell in enumerate(self.cells):
      if i in self.downsizes:
        s0 = self.reductions[r_idx](s0)
        s1 = self.reductions[r_idx+1](s1)
        r_idx += 2
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    l, r = self.proj(self.norm(s1)).split(s1.shape[1]*2, 1)
    logits = self.head(l+l*r)
    return logits

class Atten_Cell_Imagenet(nn.Module):
  def __init__(self, config, C:int, nodes:int = 6, stage:int = -1, size:int = 56, drop_path:float = 0.0):
    super(Atten_Cell_Imagenet, self).__init__()
    self.nodes = nodes
    self.C = C
    expansion_conv = config["expansion_conv"]
    expansion_mlp = 1.75
    
    print(C)
    print(drop_path)

    self.preprocess0 = ScalePerChannel(C)
    self.preprocess1 = ScalePerChannel(C)
    self.postprocess = LayerNorm2d(C, bias=False)

    self.ops = nn.ModuleList()
    for index in range(self.nodes//2):
      temp = [nn.Sequential(LayerNorm2d(C, bias=False), Attention(C, head_dim=32, num_heads=math.ceil(C/64)), DropPath(drop_path))]


      self.C_inner2 = int(C*expansion_mlp)
      temp.extend([nn.Conv2d(C, self.C_inner2, kernel_size=1, padding=0, bias=False), nn.Conv2d(self.C_inner2//2, C, kernel_size=1, padding=0, bias=False)])
      POLY_INIT_FUNC(temp[-1].weight)
      POLY_INIT_FUNC(temp[-2].weight)
      temp[-1] = nn.Sequential(LayerNorm2d(self.C_inner2//2, bias=False), temp[-1], DropPath(drop_path))
      self.ops.extend(temp)
    
    scale_vals = torch.tensor(config["sigmoid_scale"])
    self.skip_weight = torch.nn.Parameter((-scale_vals/(scale_vals - 1)).log() if config.get("paired_scale", False) else scale_vals)
    self.scale_type = config.get("paired_scale", False)
  
  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    ratios = torch.sigmoid(self.skip_weight)

    polys = self.postprocess(s0+s1)
    for i in range(self.nodes//2):
      base = i*3
      scale_idx = (i, i+1) if self.scale_type else (i*2, i*2+1)

      polys = self.ops[base](polys)*ratios[scale_idx[0]] + polys

      high, low = self.ops[base+1](polys).split(self.C_inner2//2, 1)
      polys = self.ops[base+2](high*low)*ratios[scale_idx[1]] + polys
    return polys

LowRes = {
  "nodes": [6]*8,
  "downsizes": [2, 5],
  "downsize_type": ["sep3x3", "sep3x3"],
  "channels": {2: 2, 5: 4},
  "node_norm": "layernorm",
  "expansion_conv": 1,
  "expansion_mlp": 2,
  "sigmoid_scale": [0.5] + [0.5 ** i for i in range(1, 6)]
}

class NetworkPoly(nn.Module):
  def __init__(self, C:int, num_classes:int, layers:int, config:dict, dropout:float=0, drop_path:float = 0.0):
    super(NetworkPoly, self).__init__()
    self.drop_path_prob = drop_paths(drop_path, layers)
    self.downsizes:list = config["downsizes"]

    C_prev = C_curr = C
    self.stem = nn.Conv2d(3, C_curr, 3, padding=1, bias=False)

    self.cells = nn.ModuleList()
    self.reductions = nn.ModuleList()
    stage = 0
    size = 32
    for i in range(layers):
      C_curr = int(config["channels"][i]*C) if i in config["channels"] else C_curr
      if i in self.downsizes:
        self.reductions.append(self.create_downsize(C_prev, C_curr, config["downsize_type"][0]))
        self.reductions.append(self.create_downsize(C_prev, C_curr, config["downsize_type"][1]))
        stage += 1
        size //= 2
      cell = Poly_Cell_Imagenet(config, C_curr, config["nodes"][i], stage, size, self.drop_path_prob[i])
      self.cells.append(cell)
      C_prev = C_curr

    self.norm = LayerNorm2d(C_prev, bias=False)
    self.proj = nn.Conv2d(C_prev, C_prev*4, 1, 1, 0, bias=False)
    self.head = nn.Sequential(  
      nn.AdaptiveAvgPool2d(1),
      nn.Flatten(),
      BufferedDropout(p=dropout),
      nn.Linear(C_prev*2, num_classes)
    )

  def create_downsize(self, C_prev, C_curr, down_type):
    if down_type == "sep3x3":
      layer = nn.Sequential(nn.Conv2d(C_prev, C_prev, kernel_size=3, stride=2, padding=1, groups=C_prev, bias=False),
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
      layer = nn.Sequential(nn.Conv2d(C_prev, C_prev, kernel_size=2, stride=2, groups=C_prev, bias=False),
                            nn.Conv2d(C_prev, C_curr, kernel_size=1, padding=0, bias=False))
      POLY_INIT_FUNC(layer[0].weight)
      POLY_INIT_FUNC(layer[1].weight)
    else:
      raise NotImplementedError(f"No such downsize: {down_type}")
    return layer
  
  def set_head_dropout(self, prob: float):
    p_buf = self.head[2].p
    p_buf.copy_(p_buf.new_tensor(float(prob)))

  def forward(self, input):
    s0 = s1 = self.stem(input)
    r_idx = 0
    for i, cell in enumerate(self.cells):
      if i in self.downsizes:
        s0 = self.reductions[r_idx](s0)
        s1 = self.reductions[r_idx+1](s1)
        r_idx += 2
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    l, r = self.proj(self.norm(s1)).split(s1.shape[1]*2, 1)
    logits = self.head(l+l*r)
    return logits