import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperAdapter(nn.Module):
  """
  Simple MLP Hypernet
  """
  def __init__(self, linear : nn.Module, hypernet1=None, hypernet2=None, layernorm_w_hypernet=None, layernorm_b_hypernet=None, idx=0, layer_norm_eps=1e-5):
    super().__init__()

    self.linear = linear
    self.hypernet1 = hypernet1
    self.hypernet2 = hypernet2
    self.layernorm_w_hypernet = layernorm_w_hypernet
    self.layernorm_b_hypernet = layernorm_b_hypernet

    self.idx = idx
    self.layer_norm_eps = layer_norm_eps

  def forward(self, x):
    val = self.hypernet1.dialect_features
    if self.idx is not None:
      val = nn.functional.pad(val, (0, 1), value=self.idx)
    weight1 = self.hypernet1(val)
    weight2 = self.hypernet2(val)

    layernorm_weight = self.layernorm_w_hypernet(val)
    layernorm_bias = self.layernorm_b_hypernet(val)
    x = self.linear(x)

    adapter = (F.relu(x@weight1)@weight2)
    layernorm_weight = layernorm_weight.reshape(-1)
    layernorm_bias = layernorm_bias.reshape(-1)
    out = F.layer_norm(adapter, (self.hypernet1.dim,), weight=layernorm_weight, bias=layernorm_bias, eps=self.layer_norm_eps) + x
    # out = F.relu(x@weight1)@weight2 + x
    return out

class Adapter(nn.Module):
  def __init__(self, linear : nn.Module, param=None):
    super().__init__()

    self.linear = linear
    self.adapter = nn.Parameter(torch.normal(0, 1e-7, (768, 768)))
    self.param = param
  def forward(self, x):
    x = self.linear(x)
    if self.param is not None:
      out = x@self.param
    else:
      out = x@self.adapter
    return out