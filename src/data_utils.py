import datasets
import numpy as np
from typing import Optional, List
from pandas.io.sql import execute
import torch
import pandas as pd
import os
import json
from random import sample

from datasets import interleave_datasets, load_dataset, load_metric
from multivalue.src.Dialects import *
from dataclasses import dataclass, field
from torch.utils.data import Dataset, IterableDataset
# from datasets import Dataset

from transformers import default_data_collator

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

mapping = {
    # "AbEng": AboriginalDialect(morphosyntax=True),
    "aave": AfricanAmericanVernacular(morphosyntax=True),
    "AppE": AppalachianDialect(morphosyntax=True),
    # "AusE": AustralianDialect(morphosyntax=True),
    # "AusVE": AustralianVernacular(morphosyntax=True),
    # "BahE": BahamianDialect(morphosyntax=True),
    # "BSAE": BlackSouthAfricanDialect(morphosyntax=True),
    # "CamE": CameroonDialect(morphosyntax=True),
    # "CapeE": CapeFlatsDialect(morphosyntax=True),
    # "ChIsE": ChannelIslandsDialect(morphosyntax=True),
    "ChcE": ChicanoDialect(morphosyntax=True),
    # "CollAE": ColloquialAmericanDialect(morphosyntax=True),
    "CollSgE": ColloquialSingaporeDialect(morphosyntax=True),
    # "Eaave": EarlyAfricanAmericanVernacular(morphosyntax=True),
    # "EAngE": EastAnglicanDialect(morphosyntax=True),
    # "FIE": FalklandIslandsDialect(morphosyntax=True),
    # "FijiAE": FijiAcrolect(morphosyntax=True),
    # "FijiBE": FijiBasilect(morphosyntax=True),
    # "GhanE": GhanaianDialect(morphosyntax=True),
    # "HKE": HongKongDialect(morphosyntax=True),
    "IndE": IndianDialect(morphosyntax=True),
    # "IndSAE": IndianSouthAfricanDialect(morphosyntax=True),
    # "IrishE": IrishDialect(morphosyntax=True),
    # "JamE": JamaicanDialect(morphosyntax=True),
    # "KenE": KenyanDialect(morphosyntax=True),
    # "LibE": LiberianSettlerDialect(morphosyntax=True),
    # "MalaE": MalaysianDialect(morphosyntax=True),
    # "MaltE": MalteseDialect(morphosyntax=True),
    # "ManxE": ManxDialect(morphosyntax=True),
    # "NZE": NewZealandDialect(morphosyntax=True),
    # "NflE": NewfoundlandDialect(morphosyntax=True),
    "NgE": NigerianDialect(morphosyntax=True),
    # "NEE": NorthEnglandDialect(morphosyntax=True),
    # "OSE": OrkneyShetlandDialect(morphosyntax=True),
    # "OzarkE": OzarkDialect(morphosyntax=True),
    # "PakE": PakistaniDialect(morphosyntax=True),
    # "PhilE": PhilippineDialect(morphosyntax=True),
    # "RAAE": RuralAfricanAmericanVernacular(morphosyntax=True),
    # "ScotE": ScottishDialect(morphosyntax=True),
    # "SAEE": SoutheastAmericanEnglaveDialect(morphosyntax=True),
    # "SriLE": SriLankanDialect(morphosyntax=True),
    # "StHelE": StHelenaDialect(morphosyntax=True),
    # "SEEE": SoutheastEnglandDialect(morphosyntax=True),
    # "SWEE": SouthwestEnglandDialect(morphosyntax=True),
    # "TanzE": TanzanianDialect(morphosyntax=True),
    # "TritE": TristanDialect(morphosyntax=True),
    # "UgE": UgandanDialect(morphosyntax=True),
    # "WelshE": WelshDialect(morphosyntax=True),
    # "WSAE": WhiteSouthAfricanDialect(morphosyntax=True),
    # "WZE": WhiteZimbabweanDialect(morphosyntax=True)
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    dialects: Optional[List[str]] = field(
        default=None,
        metadata={"help": "the directory where VALUE datasets will be saved"},
    )

    push_adapter_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the Adapter to HuggingFace ModelHub"},
    )

    adapter_org_id: Optional[str] = field(
        default=None,
        metadata={"help": "Organization to contain AdapterHub repo"},
    )

    adapter_repo_id: Optional[str] = field(
        default=None,
        metadata={"help": "the Hub Repo name for model to push"},
    )

    combine_sae: bool = field(
        default=False,
        metadata={
            "help": "Combine Training Data from SAE and Selected Dialect Transform"
        },
    )

    text_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets to do alignment on"},
    )

    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )

    save_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "Save augmented dialect data"}
    )

    save_dataset_path: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Path to save augmented dialect data"}
    )

    load_dataset: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, does not preprocess dialect data, only loads from save_dataset_path"}
    )

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The data directory containing input files."}
    )
    
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )

    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )

    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )

    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(task_to_keys.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

class Preprocessing:
    
    def __init__(self, sentence1_key, sentence2_key, tokenizer, padding, max_seq_length, label_to_id):
        self.sentence1_key = sentence1_key
        self.sentence2_key = sentence2_key
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_seq_length = max_seq_length
        self.label_to_id = label_to_id

    def process_rules(self, executed_rules):
        with open("resources/feature_id_to_function_name.json", "r") as file:
            id_map = json.load(file)
            inv_map = {}
            for k, v in id_map.items():
                for type in v:
                    inv_map[type] = int(k)
            rules = []
            for _, v in executed_rules.items():
                for t, n in v.items():
                    if t == "type":
                        rules.append(inv_map[n])

            feature_vec = np.zeros(236)
            for r in rules:
                feature_vec[r] = 1.
            return feature_vec

    def dialect_transform_factory(self, dialect_name):
        def dialect_transform(examples):
            dialect = None
            # mapping = {}
            # if dialect_name == "aave":
            #     dialect = AfricanAmericanVernacular(morphosyntax=True)
            # elif dialect_name == "IndE":
            #     dialect = IndianDialect(morphosyntax=True)
            # elif dialect_name == "ChcE":
            #     dialect = ChicanoDialect(morphosyntax=True)
            # elif dialect_name == "HK":
            #     dialect = HongKongDialect(morphosyntax=True)
            # elif dialect_name == "Malay":
            #     dialect = MalaysianDialect(morphosyntax=True)
            # elif dialect_name == "NgE":
            #     dialect = NigerianDialect(morphosyntax=True)
            # elif dialect_name == "singapore":
            #     dialect = ColloquialSingaporeDialect(morphosyntax=True)
            dialect = mapping[dialect_name]

            # print(dialect)
            if dialect:
                # conversions1 = [
                #     dialect.convert_sae_to_dialect(example)
                #     for example in examples[self.sentence1_key]
                # ]
                conversions1 = []
                executed_rules = []
                for example in examples[self.sentence1_key]:
                    conversions1.append(dialect.convert_sae_to_dialect(example))
                    processed_rules = self.process_rules(dialect.executed_rules)
                    executed_rules.append(processed_rules)
                examples[self.sentence1_key] = conversions1
                examples["dialect_features"] = [dialect.vector for i in range(len(conversions1))]
                examples["dialect_name"] = [dialect_name for i in range(len(conversions1))]
                if self.sentence2_key is None:
                    return examples
                else:
                    conversions2 = [
                        dialect.convert_sae_to_dialect(example)
                        for example in examples[self.sentence2_key]
                    ]
                    examples[self.sentence2_key] = conversions2
                    return examples
            else:
                return examples
        return dialect_transform

    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],)
            if self.sentence2_key is None
            else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = self.tokenizer(
            *args, padding=self.padding, max_length=self.max_seq_length, truncation=True
        )

        # Map labels to IDs (not necessary for GLUE tasks)
        if self.label_to_id is not None and "label" in examples:
            result["label"] = [
                (self.label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]
        if "dialect_features" in examples:
            result["dialect_features"] = examples["dialect_features"]
        else:
            result["dialect_features"] = torch.zeros(examples["dialect_features"].shape)
        if "dialect_name" in examples:
            result["dialect_name"] = examples["dialect_name"]
        else:
            result["dialect_name"] = None
        # if "executed_features" in examples:
        #     result["executed_features"] = examples["executed_features"]
        # else:
        #     result["executed_features"] = torch.zeros(examples["executed_features"].shape)

        return result
    
    def format_function(self, examples):
        args = (
            (examples[self.sentence1_key],)
            if self.sentence2_key is None
            else (examples[self.sentence1_key], examples[self.sentence2_key])
        )
        result = {"args": args}

        # Map labels to IDs (not necessary for GLUE tasks)
        if self.label_to_id is not None and "label" in examples:
            result["label"] = [
                (self.label_to_id[l] if l != -1 else -1) for l in examples["label"]
            ]

        return result

def dialect_data_collator(features):
    dialect_feats = []
    idx = set()
    for f in features:
        if f["dialect_features"] is None:
            f["dialect_features"] = [0]*236
        dialect_feats.append(torch.Tensor(f["dialect_features"][1:]))
        idx.add(f["dialect_features"][-1])
        del f["dialect_features"]

    # for i, f in enumerate(features):
    #     if "label" in f.keys():
    #         features[i]["labels"] = f["label"]
    batch = default_data_collator(features)

    batch["dialect_features"] = dialect_feats[0]

    assert len(idx) == 1, "MORE THAN 1 DATASET IN BATCH"
    return batch

def label_data_collator(features):
    # for i, f in enumerate(features):
    #     if "label" in f.keys():
    #         features[i]["labels"] = f["label"]
    batch = default_data_collator(features)
    return batch

def group_data_collator(features):
    dialect_feats = []
    idx = set()
    dialect_name = None
    for f in features:
        if f["dialect_features"] is None:
            f["dialect_features"] = [0]*236
        dialect_feats.append(torch.Tensor(f["dialect_features"][1:]))
        dialect_name = f["dialect_name"]
        idx.add(f["dialect_features"][-1])
        del f["dialect_features"]

    # for i, f in enumerate(features):
    #     if "label" in f.keys():
    #         features[i]["labels"] = f["label"]
    batch = default_data_collator(features)
    batch["dialect_name"] = dialect_name
    batch["dialect_features"] = dialect_feats[0]
    
    assert len(idx) == 1, "MORE THAN 1 DATASET IN BATCH"
    return batch