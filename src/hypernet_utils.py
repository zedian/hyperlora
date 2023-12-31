import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class HyperNet(nn.Module):
  def __init__(self, input_dim: int, batch_size: int, mlp_dim: int, num_embeddings: int, task_dim : int= None, layer_dim : int = None):
    super().__init__()
    self.input_dim = input_dim
    self.mlp_dim = mlp_dim
    self.batch_size = batch_size
    self.num_embeddings = num_embeddings
    self.transformer_encoder = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=mlp_dim, batch_first=True)
    self.transformer = nn.TransformerEncoder(self.transformer_encoder, num_layers=1)
    self.weights = nn.Parameter(torch.zeros(batch_size, num_embeddings, input_dim))

    self.weights = nn.init.kaiming_uniform_(self.weights)
  def forward(self, x, task_emb = None, layer_emb = None):
    rep = torch.cat((x, self.weights), dim=1)
    output = self.transformer(rep)
    output = output[:, 128:, :]
    print(output.shape)
    return output

class HyperParamNet(nn.Module):
  """
  Helper class to wrap together hypernetwork weights "linear1" and "linear2" into MLP

  Arguments:
  - linear1 : Downsampling weight (encoding dim x bottleneck)
  - linear2 : Upsampling weight (bottleneck x dim)
  - dim : output dim
  - bottleneck : bottleneck dim

  Output:
  Adapter weight generated by hypernetworks with dialect feature input
  """
  def __init__(self, linear1, linear2, dim, bottleneck, scale=False):
    super().__init__()
    self.linear1 = linear1
    self.linear2 = linear2
    self.dim = dim #Output dimension
    self.bottleneck = bottleneck #MLP bottleneck
    if scale:
      self.scale = math.sqrt(dim)
    else:
      self.scale = 1

  def set_dialect_features(self, dialect_features):
    self.dialect_features = dialect_features
    return
  
  def forward(self, dialect_features):
    output = self.linear2(F.relu(self.linear1(dialect_features))).reshape(self.dim, self.bottleneck)
    return output/self.scale

class HyperParamDeepNet(nn.Module):
  """
  Helper class to wrap together hypernetwork weights "linear1" and "linear2" into MLP

  Arguments:
  - linear1 : Downsampling weight (encoding dim x bottleneck)
  - linear2 : Upsampling weight (bottleneck x dim)
  - dim : output dim
  - bottleneck : bottleneck dim

  Output:
  Adapter weight generated by hypernetworks with dialect feature input
  """
  def __init__(self, linear1, linear2, middle1, middle2, dim, bottleneck, scale=False):
    super().__init__()
    self.linear1 = linear1
    self.linear2 = linear2
    self.middle1 = middle1
    self.middle2 = middle2
    self.dim = dim #Output dimension
    self.bottleneck = bottleneck #MLP bottleneck
    self.layernorm = nn.LayerNorm(64, eps=1e-7)
    if scale:
      self.scale = math.sqrt(dim)
    else:
      self.scale = 1

  def set_dialect_features(self, dialect_features):
    self.dialect_features = dialect_features
    return
  
  def forward(self, dialect_features):
    h = F.relu(self.linear1(dialect_features))
    h = self.middle2(F.relu(self.middle1(self.layernorm(h)))) + h
    output = self.linear2(h).reshape(self.dim, self.bottleneck)
    return output/self.scale

class HyperParamSelfAttn(nn.Module):
  """
  Helper class to wrap together hypernetwork weights "linear1" and "linear2" into MLP

  Arguments:
  - linear1 : Downsampling weight (encoding dim x bottleneck)
  - linear2 : Upsampling weight (bottleneck x dim)
  - dim : output dim
  - bottleneck : bottleneck dim

  Output:
  Adapter weight generated by hypernetworks with dialect feature input
  """
  def __init__(self, linear1, linear2, query, key, value, dropout, dim, bottleneck):
    super().__init__()
    self.linear1 = linear1
    self.linear2 = linear2
    self.query = query
    self.key = key
    self.value = value
    self.dropout = nn.Dropout(p=dropout)
    self.dim = dim
    self.bottleneck = bottleneck

  def set_dialect_features(self, dialect_features):
    self.dialect_features = dialect_features
    return
  
  def forward(self, dialect_features):
    h = F.relu(self.linear1(dialect_features))

    Q = self.query(h)
    K = self.key(h)
    V = self.value(h)

    output = scaled_dpa(Q, K, V, dropout=self.dropout) + h
    output = self.linear2(output).reshape(self.dim, self.bottleneck)
    return output

def scaled_dpa(Q,K,V,dropout):
  scale_factor = 1 / math.sqrt(Q.size(-1))
  attn_weight = torch.softmax(torch.outer(Q, K)* scale_factor, dim=-1)
  attn_weight = dropout(attn_weight)
  return attn_weight @ V