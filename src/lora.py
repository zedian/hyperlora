import torch
import torch.nn as nn
import torch.nn.functional as F

from torchreparam import ReparamModule
from functorch import vmap

class LoRAWrapper(nn.Module):
    def __init__(self, conv1dmodule: nn.Module, lora_rank: int):
        super().__init__()

        self.base_module = conv1dmodule

        d1, d2 = self.base_module.weight.shape
        self.A = nn.Parameter(torch.zeros((d2, lora_rank)))
        self.B = nn.Parameter(torch.normal(0, 1, size=(d1, lora_rank)))

    def forward(self, x, a = None, b = None):

        # HyperLoRA
        if a and b:
            return (x@b)@a.T + x@self.base_module.weight
        return (x@ self.B)@ self.A.T + x@self.base_module.weight

class HyperReparamLoRA(nn.Module):
  def __init__(self, linear : nn.Module, hypernet_A=None, hypernet_B=None):
    super().__init__()

    self.weight = linear.weight
    self.bias = linear.bias
    self.lora_A = hypernet_A
    self.lora_B = hypernet_B

  def forward(self, x):

    # Generate A weights
    val = self.lora_A.dialect_features
    lora_A_weight = self.lora_A(val)
    lora_B_weight = self.lora_B(val)
    out = x@self.weight + self.bias + (x@ lora_A_weight)@lora_B_weight
    return out

class HyperLora(nn.Module):
  """
  Simple MLP Hypernet
  """
  def __init__(self, linear : nn.Module, hypernet1=None, hypernet2=None, idx=0):
    super().__init__()

    self.linear = linear
    self.hypernet1 = hypernet1
    self.hypernet2 = hypernet2
    self.idx = idx

  def forward(self, x):
    val = self.hypernet1.dialect_features
    if self.idx is not None:
      val[-1] = self.idx
    weight1 = self.hypernet1(val)
    weight2 = self.hypernet2(val)
    
    out = self.linear(x)
    out = (x@weight1)@weight2 + out
    return out
