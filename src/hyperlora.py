import torch.nn as nn
import torch.nn.functional as F
import torch
import pickle
from torchtyping import TensorType
from typing import Optional

DEVICE=""
class HyperParamNet(nn.Module):
  """
  Helper class to wrap together hypernetwork weights "linear1" and "linear2" into MLP

  Arguments:
  - linear1 : Downsampling weight (encoding dim x bottleneck)
  - linear2 : Upsampling weight (bottleneck x dim)
  - dim : output dim
  - bottleneck : bottleneck dim

  Output:
  Main weight generated by hypernetworks with dialect feature input
  """
  def __init__(self, linear1, linear2, dim, bottleneck):
    super().__init__()
    self.linear1 = linear1
    self.linear2 = linear2
    self.dim = dim #Output dimension
    self.bottleneck = bottleneck #MLP bottleneck

  def set_condition_var(self, condition_var):
    self.condition_var = condition_var
    return
  
  def forward(self, condition_var):
    output = self.linear2(F.relu(self.linear1(condition_var))).reshape(self.dim, self.bottleneck)
    return output

class HyperNet(nn.Module):
    def __init__(self, encoding_dim, input_dim, embedding_dim):
        super(HyperNet, self).__init__()

        # Define hypernet (Here its just a 1 layer net)
        self.pre_down_linear = nn.Identity()
        self.down_linear = nn.Linear(encoding_dim+1, input_dim*embedding_dim)
        self.down_linear.weight, self.down_linear.bias = self.init_layer(self.down_linear)
        self.pre_up_linear = nn.Identity()
        self.up_linear = nn.Linear(encoding_dim+1, embedding_dim*input_dim)
        self.up_linear.weight, self.up_linear.bias = self.init_layer(self.up_linear)

        self.down_hypernet = HyperParamNet(self.pre_down_linear, self.down_linear, input_dim, embedding_dim)
        self.up_hypernet = HyperParamNet(self.pre_up_linear, self.up_linear, embedding_dim, input_dim)

    def init_layer(self, layer):
        # Initialize hypernet weights
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        bias = nn.init.zeros_(layer.bias)
        return weight, bias

class HyperLora(nn.Module):
  """
  Simple MLP Hypernet
  """
  def __init__(self, linear : nn.Module, hypernet1=None, hypernet2=None, idx=0):
    super().__init__()

    self.linear = linear
    self.hypernet1 = hypernet1
    self.hypernet2 = hypernet2
    self.dropout = nn.Dropout(p=0.1)
    # Layer idx
    self.idx = idx

  def forward(self, x):
    # Conditioning variable (either indicator or example)
    val = self.hypernet1.dialect_features
    # Layer idx is added to conditioning variable
    if self.idx is not None:
      val = nn.functional.pad(val, (0, 1), value=self.idx)

    # Compute hypernet weights
    weight1 = self.hypernet1(val)
    weight2 = self.hypernet2(val)

    # Apply lora
    out = self.linear(x)
    out = (x@weight1)@weight2 + out
    return out

class AdapterWrapper(nn.Module):
    """
    General Wrapper Class for Hypernet Config

    Each child class needs to implement the init hypernet method that injects hypernet weights
    """
    def __init__(self, model, dialects, embedding_dim, input_dim, weights):
        super(AdapterWrapper, self).__init__()
        self.model = model
        self.down_hypernet = None
        self.up_hypernet = None
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.encoding_dim = 8
        self.dialects = dialects

        self.hypernet = HyperNet(self.encoding_dim, self.input_dim, self.embedding_dim)

        if weights is not None:
            self.hypernet.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
            print("WEIGHTS LOADED")

        self.init_hypernet()

    def init_layer(self, layer):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        bias = nn.init.zeros_(layer.bias)
        return weight, bias

    def init_hypernet(self):
        pass

    def freeze_params(self):
        for layer in self.model.modules():
            for _, param in layer.named_parameters():
                param.requires_grad = False

    def forward(self, labels, input_ids, attention_mask, condition_var, **kwargs):
        """
        forward model needs to include dialect_features parameter for Trainer to not discard this feature
        """
        inputs = {"labels":labels, "input_ids": input_ids, "attention_mask": attention_mask, **kwargs}

        # Set conditioning variables
        self.hypernet.down_hypernet.set_condition_var(condition_var)
        self.hypernet.up_hypernet.set_condition_var(condition_var)
        outputs = self.model(**inputs)

        return outputs