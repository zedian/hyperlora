from cProfile import label
from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
import torch
import numpy as np
import pickle
from transformers import RobertaForSequenceClassification
from torchtyping import TensorType
from transformers.utils.dummy_pt_objects import AutoModelForSequenceClassification, RobertaModel

from geomloss import SamplesLoss

from transformers.adapters import (
    RobertaAdapterModel,
)

from src.hypernet_utils import HyperParamNet, HyperParamSelfAttn, HyperParamDeepNet
from src.hyperadapter import HyperAdapter, Adapter
from src.hyperlora import HyperLora

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PCA_EMB_FILE = "pca4.pkl"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={
            "help": "Will enable to load a pretrained model whose head dimensions are different."
        },
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    apply_hyperlora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "LoRA alpha"},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={
            "help": "Rank of LoRA updates"
        }
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply Adapter or not."},
    )
    apply_hyperadapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply HyperAdapter or not."},
    )
    task_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Add task adapter"}
    )
    eval_dialect: Optional[str] = field(
        default="sae",
        metadata={"help": "Single dialect evaluation"}
    )
    hidden_adapter_dim: Optional[int] = field(
        default=768,
        metadata={"help": "Input dimension of adapter"}
    )
    adapter_emb_dim: Optional[int] = field(
        default=48,
        metadata={"help": "Embedding dimension of adapter"}
    )
    load_hypernet_weights: Optional[str] = field(
        default=None,
        metadata={"help": "Path to hypernet weights, otherwise random init"}
    )
    groupdro: Optional[bool] = field(
        default=False,
        metadata={"help": "Robust training"}
    )

class HyperNet(nn.Module):
    def __init__(self, encoding_dim, input_dim, embedding_dim):
        super(HyperNet, self).__init__()
        self.output_dim = 768
        self.hidden_dim = 8
        self.pre_down_linear = nn.Linear(encoding_dim+1, self.hidden_dim)
        self.pre_down_linear.weight, self.pre_down_linear.bias = self.init_layer(self.pre_down_linear)
        self.down_linear = nn.Linear(self.hidden_dim, input_dim*embedding_dim)
        self.down_linear.weight, self.down_linear.bias = self.init_layer(self.down_linear)
        self.pre_up_linear = nn.Linear(encoding_dim+1, self.hidden_dim)
        self.pre_up_linear.weight, self.pre_up_linear.bias = self.init_layer(self.pre_up_linear)
        self.up_linear = nn.Linear(self.hidden_dim, embedding_dim*self.output_dim)
        self.up_linear.weight, self.up_linear.bias = self.init_layer(self.up_linear)

        # self.pre_layer_norm_w = nn.Linear(encoding_dim+1, self.hidden_dim, bias=False)
        # self.pre_layer_norm_w.weight, _ = self.init_layer(self.pre_layer_norm_w, bias=False)
        # self.layer_norm_w = nn.Linear(encoding_dim+1, self.output_dim, bias=False)
        # self.layer_norm_w.weight, _ = self.init_layer(self.layer_norm_w, bias=False)

        # self.pre_layer_norm_b = nn.Linear(encoding_dim+1, self.hidden_dim, bias=False)
        # self.pre_layer_norm_b.weight, _ = self.init_layer(self.pre_layer_norm_b, bias=False)
        # self.layer_norm_b = nn.Linear(encoding_dim+1, self.output_dim, bias=False)
        # self.layer_norm_b.weight, _ = self.init_layer(self.layer_norm_b, bias=False)

        self.down_hypernet = HyperParamNet(self.pre_down_linear, self.down_linear, input_dim, embedding_dim)
        self.up_hypernet = HyperParamNet(self.pre_up_linear, self.up_linear, embedding_dim, self.output_dim, scale=True)
        # self.layernorm_w_hypernet = HyperParamNet(nn.Identity(), self.layer_norm_w, self.output_dim, 1)
        # self.layernorm_b_hypernet = HyperParamNet(nn.Identity(), self.layer_norm_b, self.output_dim, 1)

    def init_layer(self, layer, bias=True):
        weight = nn.Parameter(torch.normal(0, 1e-7, layer.weight.shape))
        if bias:
            bias = nn.init.zeros_(layer.bias)
        else:
            bias = None
        return weight, bias

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
        self.encoding_dim = 235
        self.dialects = dialects

        self.hypernet = HyperNet(self.encoding_dim, self.input_dim, self.embedding_dim)

        if weights is not None:
            self.hypernet.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
            print("WEIGHTS LOADED")

        self.earth_mover_loss = SamplesLoss(loss="sinkhorn", p=2)

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

    @torch.no_grad()
    def produce_original_embeddings(
        self,
        input_ids: TensorType["batch", "seq_len"],
        attention_mask: TensorType["batch", "seq_len"],
        dialect_features: TensorType["batch", "seq_len"],
        token_type_ids: Optional[TensorType["batch", "seq_len"]] = None,
        position_ids: Optional[TensorType["batch", "seq_len"]] = None,
        head_mask: Optional[TensorType["layers", "heads"]] = None,
    ) -> TensorType["batch", "seq_len", "hidden_size"]:
        self.train(False)
        outputs = self.last_emb(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            dialect_features=dialect_features,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            include_original=False
        )

        self.train(True)
        return outputs.last_hidden_state, attention_mask

    def get_weight(self, mask):             
        probs = torch.div(mask, mask.sum(1).reshape(-1,1))                                                                                                          
        return probs
  
    def emb(self, l):
        """
        PCA Embedding of linguistic attestation vector
        """
        feature = l
        if not isinstance(l, torch.Tensor):
            feature = torch.Tensor(l)
        return feature.to(DEVICE)


    def last_emb(self, input_ids, attention_mask, dialect_features, original_mask=None, original_embedding=None, include_original=True,**kwargs):
        """
        forward model needs to include dialect_features parameter for Trainer to not discard this feature
        """
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        self.hypernet.down_hypernet.set_dialect_features(self.emb(dialect_features))
        self.hypernet.up_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_w_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_b_hypernet.set_dialect_features(self.emb(dialect_features))
        outputs = self.model(**inputs)
        return outputs

    def forward(self, labels, input_ids, attention_mask, dialect_features, original_mask=None, original_embedding=None, include_original=False, **kwargs):
        """
        forward model needs to include dialect_features parameter for Trainer to not discard this feature
        """
        inputs = {"labels":labels, "input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        if include_original:
            inputs["original_embedding"] = original_embedding
            inputs["original_mask"] = original_mask

        self.hypernet.down_hypernet.set_dialect_features(self.emb(dialect_features))
        self.hypernet.up_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_w_hypernet.set_dialect_features(self.emb(dialect_features))
        # self.hypernet.layernorm_b_hypernet.set_dialect_features(self.emb(dialect_features))
        outputs = self.model(**inputs)
        
        # if "last_hidden_state" in outputs:
        #     hidden_mat = outputs.last_hidden_state
        # else:
        #     hidden_mat = outputs.encoder_last_hidden_state

        # alignment_loss = self.earth_mover_loss(                                                                                                                                
        #     self.get_weight(attention_mask), hidden_mat, self.get_weight(original_mask.reshape(attention_mask.shape)), original_embedding                                                                                       
        # )

        return outputs

class RobertaLoraWrapper(AdapterWrapper):
    def __init__(self, model, dialects, embedding_dim, input_dim, weights):
        super().__init__(model, dialects, embedding_dim, input_dim, weights)
    
    def init_hypernet(self):
        for i, l in enumerate(self.model.roberta.encoder.layer):
            l.attention.self.query = HyperLora(l.attention.self.query, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i)
            l.attention.self.value = HyperLora(l.attention.self.value, self.hypernet.down_hypernet, self.hypernet.up_hypernet, 2*i+1)

        self.freeze_params()
        self.hypernet.pre_down_linear.weight.requires_grad = True
        self.hypernet.pre_down_linear.bias.requires_grad = True
        self.hypernet.pre_up_linear.weight.requires_grad = True
        self.hypernet.pre_up_linear.bias.requires_grad = True
        self.hypernet.down_linear.weight.requires_grad = True
        self.hypernet.down_linear.bias.requires_grad = True
        self.hypernet.up_linear.weight.requires_grad = True
        self.hypernet.up_linear.bias.requires_grad = True

class TaskRobertaAdapterWrapper(AdapterWrapper):
    def __init__(self, model, dialects, input_dim):
        super(TaskRobertaAdapterWrapper, self).__init__(model, dialects, input_dim)
    
    def init_hypernet(self):
        self.down_hypernet = HyperParamNet(self.pre_down_linear, self.down_linear, self.input_dim, self.embedding_dim)
        self.up_hypernet = HyperParamNet(self.pre_up_linear, self.up_linear, self.embedding_dim, self.input_dim)
        for i, l in enumerate(self.model.roberta.encoder.layer):
            l.output.dense = HyperAdapter(l.output.dense, self.down_hypernet, self.up_hypernet, i)
        
        self.freeze_params()
        self.down_linear.weight.requires_grad = True
        self.down_linear.bias.requires_grad = True
        self.up_linear.weight.requires_grad = True
        self.up_linear.bias.requires_grad = True

class T5AdapterWrapper(AdapterWrapper):
    def __init__(self, model, dialects, input_dim):
        super(T5AdapterWrapper, self).__init__(model, dialects, input_dim)

    def init_hypernet(self):
        self.down_hypernet = HyperParamNet(self.pre_down_linear, self.down_linear, self.input_dim, self.embedding_dim)
        self.up_hypernet = HyperParamNet(self.pre_up_linear, self.up_linear, self.embedding_dim, self.input_dim)
        for i, l in enumerate(self.model.encoder.block):
            l.layer[0].SelfAttention.o = HyperAdapter(l.layer[0].SelfAttention.o, self.down_hypernet, self.up_hypernet, i)
            # l.layer[1].DenseReluDense.wo = HyperAdapter(l.layer[1].DenseReluDense.wo, self.hypernet1, self.hypernet2, i)
        
        self.freeze_params()
        self.down_linear.weight.requires_grad = True
        self.down_linear.bias.requires_grad = True
        self.up_linear.weight.requires_grad = True
        self.up_linear.bias.requires_grad = True

class TaskAwareWrapper(nn.Module):
    def __init__(self, model, dialects):
        super(TaskAwareWrapper, self).__init__()
        self.model = model
        self.hypernet = None
        self.dialect_to_embeddings = nn.ParameterDict(dict())
        self.embedding_dim = 256
        self.dialects = dialects
        
        # self.init_dict(self.dialects)
        self.add_adapters()

    def add_adapters(self):
      for l in self.model.roberta.encoder.layer:
        param = nn.Parameter(torch.normal(0, 1e-7, (768, 768)))
        l.output.dense = Adapter(l.output.dense, param)
        # l.output.dense = Adapter(l.output.dense)
      for layer in self.model.modules():
        if not isinstance(layer, Adapter):
          for name, param in layer.named_parameters():
            param.requires_grad = False
        else:
          for name, param in layer.named_parameters():
            if "param" in name:
            # if "adapter" in name:
              param.requires_grad = True
            else:
              param.requires_grad = False

    def init_dict(self, dialects):
        for dialect in dialects:
            dialect_embedding = torch.Tensor(torch.randn(self.embedding_dim))
            self.dialect_to_embeddings[dialect] = nn.Parameter(dialect_embedding)

    def emb(self, l):
        if torch.count_nonzero(l) == 0:
            return torch.zeros(self.embedding_dim) + 0
        else:
            feature_vec = nn.functional.pad(l, (0, 1))
            return torch.zeros(self.embedding_dim) + feature_vec

    def forward(self, labels, input_ids, attention_mask, dialect_features, executed_features, **kwargs):
        inputs = {"labels": labels, "input_ids": input_ids, "attention_mask": attention_mask}
        output = self.model(**inputs)
        return output

class GroupAdapterWrapper(nn.Module):
    """
    General Wrapper Class for Hypernet Config

    Each child class needs to implement the init hypernet method that injects hypernet weights
    """
    def __init__(self, model, dialects, step_size):
        super(GroupAdapterWrapper, self).__init__()
        self.model = model
        self.n_groups = len(dialects)
        self.step_size = step_size

        self.idx_group = dict(zip(dialects, list(range(self.n_groups))))
        self.idx_group["sae"] = self.n_groups
        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups + 1)/(self.n_groups + 1)

    def forward(self, labels, input_ids, attention_mask, dialect_name, **kwargs):
        """
        forward model needs to include dialect_features parameter for Trainer to not discard this feature
        """
        inputs = {"labels": labels, "input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        
        output = self.model(**inputs)

        if dialect_name is None:
            dialect_name = "sae"
        groupidx = self.idx_group[dialect_name]
        groupidx
        nu = torch.FloatTensor([self.step_size])

        cloneloss = output.loss.clone()

        probabilities = []
        for i, p in enumerate(self.adv_probs):
            if i != groupidx:
                probabilities.append(p)
            else:
                probabilities.append(p * torch.exp(self.step_size * cloneloss))
        self.adv_probs = torch.FloatTensor(probabilities)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = cloneloss * self.adv_probs[groupidx]
        return (robust_loss,)
