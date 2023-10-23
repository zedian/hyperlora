#!/usr/bin/env python
# coding=utf-8

"""
Aligning Embeddings using contrastive learning on aligned data
"""

import logging
import os
import sys
import json
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset

import transformers
import datasets
from datasets import concatenate_datasets, load_dataset, load_metric
from transformers.adapters.modeling import Adapter

from transformers import (
    AutoModel,
    AutoConfig,
    AdapterConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    AdapterTrainer,
    AutoAdapterModel,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    LoRAConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

import pickle as pkl

from src.data_utils import DataTrainingArguments, Preprocessing, dialect_data_collator
from src.model_utils import ModelArguments, RobertaLoraWrapper
from src.utils import task_to_keys

from src.trainer_utils import RobertaTrainer

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to
# the sys.path.
sys.path.append(parent)
from multivalue.src.Dialects import *

logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0")

def collate_fn(examples):
    original_embedding = torch.stack(
        [
            torch.tensor(example["original_embedding"]).reshape(128, -1)
            for example in examples
        ]
    )

    original_mask = torch.stack(
        [
            torch.tensor(example["original_mask"]).reshape(128, -1)
            for example in examples
        ]
    )

    dialect_feats = []

    for f in examples:
        if f["dialect_features"] is None:
            f["dialect_features"] = [0]*236
        dialect_feats.append(torch.Tensor(f["dialect_features"][1:]))
        del f["dialect_features"]

    input_ids = torch.tensor(
        [example["input_ids"] for example in examples], dtype=torch.long
    )
    attention_mask = torch.tensor(
        [example["attention_mask"] for example in examples], dtype=torch.long
    )
    return {
        "original_embedding": original_embedding,
        "original_mask": original_mask,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "dialect_features": dialect_feats[0],
        "labels": torch.zeros(len(examples))
    }


def main():
    # 1. Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
        )
    )
    (
        model_args,
        data_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 3. Detecting last checkpoint and eventualy continue from last checkpoint
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # 4. Load dataset
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full image path and the second column for the
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            data_dir=data_args.data_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # 5. Load pretrained model, tokenizer, and feature extractor

    ConfigClass = AutoConfig
    TokenizerClass = AutoTokenizer
    if model_args.task_adapter:
        ModelClass = AutoAdapterModel
    else:
        ModelClass = AutoModelForSequenceClassification

    if model_args.tokenizer_name:
        tokenizer = TokenizerClass.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    elif model_args.model_name_or_path:
        tokenizer = TokenizerClass.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "roberta" in model_args.model_name_or_path:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    config = model.config

    if model_args.apply_hyperlora:
        model = RobertaLoraWrapper(model, data_args.dialects, model_args.lora_rank, model_args.hidden_adapter_dim, model_args.load_hypernet_weights, train=True)

    if hasattr(model, "model"):
        innermodel = model.model
    else:
        innermodel = model

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    text_column = data_args.text_column
    if text_column not in column_names:
        raise ValueError(
            f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
        )

    transformed_column = "transformed_" + text_column

    all_datasets = {k: None for k in data_args.dialects}

    def dialect_transform_factory(dialect_name):
        def dialect_transform(examples):
            dialect = None
            if dialect_name == "aave":
                dialect = AfricanAmericanVernacular(morphosyntax=True)
            elif dialect_name == "AppE":
                dialect = AppalachianDialect(morphosyntax=True)
            elif dialect_name == "IndianEnglish":
                dialect = IndianDialect(morphosyntax=True)
            elif dialect_name == "CollSgE":
                dialect = ColloquialSingaporeDialect(morphosyntax=True)
            elif dialect_name == "NigerianEnglish":
                dialect = NigerianDialect(morphosyntax=True)
            elif dialect_name == "ChcE":
                dialect = ChicanoDialect(morphosyntax=True)

            original_text = examples[text_column] + examples["sentence2"]
            transformed_text = [
                dialect.convert_sae_to_dialect(example) for example in original_text
            ]
            del examples
            examples = {}
            examples["original"] = original_text
            examples["transformed"] = transformed_text
            examples["dialect_features"] = [dialect.vector for i in range(len(transformed_text))]

            return examples

        return dialect_transform

    for i, dialect in enumerate(data_args.dialects):
        dialect_transform = dialect_transform_factory(dialect)
        
        with training_args.main_process_first(desc="dataset map pre-processing"):
            if not data_args.load_dataset:
                all_datasets[dialect] = dataset.map(
                    dialect_transform,
                    batched=True,
                    remove_columns=[
                        column_name
                        for column_name in column_names
                        if column_name not in ["original", "transformed"]
                    ],
                    load_from_cache_file=not data_args.overwrite_cache,
                    num_proc=4,
                    desc="Transform Dataset Using Dialect Transformations",
                )
            else:
                all_datasets[dialect] = datasets.load_from_disk(data_args.save_dataset_path[i])
            if data_args.save_dataset:
                all_datasets[dialect].save_to_disk(data_args.save_dataset_path[i])

        if training_args.do_train:
            if "train" not in dataset:
                raise ValueError("--do_train requires a train dataset")

            if data_args.max_train_samples is not None:
                max_train_samples = min(len(dataset["train"]), data_args.max_train_samples)
                dataset["train"] = dataset["train"].select(range(max_train_samples))
                all_datasets[dialect]["train"] = all_datasets[dialect]["train"].select(range(max_train_samples))

        # NOTE: Should we even eval on all dialects?
        # Possible solution: compute dialect specific accuracy
        if training_args.do_eval:
            if (
                "validation" not in dataset
                and "validation_matched" not in dataset
            ):
                raise ValueError("--do_eval requires a validation dataset")

            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(dataset["validation"]), data_args.max_eval_samples)
                dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
                all_datasets[dialect]["validation"] = all_datasets[dialect]["validation"].select(range(max_eval_samples))

        if (
            training_args.do_predict
            or data_args.task_name is not None
            or data_args.test_file is not None
        ):
            if "test" not in dataset and "test_matched" not in dataset:
                raise ValueError("--do_predict requires a test dataset")
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(dataset["test"]), data_args.max_predict_samples
            )
            dataset["test"] = dataset["test"].select(range(max_predict_samples))
            all_datasets[dialect]["test"] = all_datasets[dialect]["test"].select(range(max_predict_samples))

    if data_args.combine_sae:
        all_datasets["sae"] = dataset
        combine_on = "sae"
    else:
        dataset["train"] = all_datasets[data_args.dialects[0]]["train"].select([])
        dataset["validation"] = all_datasets[data_args.dialects[0]]["validation"].select([])
        dataset["test"] = all_datasets[data_args.dialects[0]]["test"].select([])
        all_datasets["sae"] = dataset
        combine_on = "sae"

    for split in dataset:
        dialect_list = []

        if data_args.combine_sae:
            dialect_list.append(all_datasets["sae"][split])
        for i, dialect in enumerate(data_args.dialects):
            dialect_list.append(all_datasets[dialect][split])
        all_datasets[combine_on][split] = concatenate_datasets(dialect_list)

    dialect_datasets = all_datasets[combine_on]
    dataset_sizes = [len(all_datasets[data_args.dialects[0]]["train"]) for _ in [*data_args.dialects]]
    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_text(examples):
        result = tokenizer(
            examples["transformed"],
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        if "dialect_features" in examples:
            result["dialect_features"] = examples["dialect_features"]
        return result

    # Generate Fixed Original Embeddings
    def embed_text(examples):
        result = tokenizer(
            examples["original"],
            max_length=data_args.max_seq_length,
            padding="max_length",
            truncation=True,
        )

        batch_size = 128
        original_embeddings = []
        original_mask = []
        for i in range(0, len(result["input_ids"]), batch_size):
            input_ids = torch.tensor(result["input_ids"][i : i + batch_size])
            attention_mask = torch.tensor(
                result["attention_mask"][i : i + batch_size]
            )
            dialect_features = examples["dialect_features"][i]
            hidden_mat, attn_mask = model.produce_original_embeddings(input_ids, attention_mask, dialect_features=dialect_features)
            original_embeddings.extend(
                [embedding.flatten() for embedding in torch.split(hidden_mat, 1)]
            )
            original_mask.extend(
                [mask.flatten() for mask in torch.split(attn_mask, 1)]
            )
        examples["original_embedding"] = original_embeddings
        examples["original_mask"] = original_mask

        return examples

    model = model
    dataset = dialect_datasets.map(
        tokenize_text,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    ).map(
        embed_text,
        batched=True,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Original Embeddings for Untransformed Input",
    )

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return {"mean_loss":preds.mean()}

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    predict_dataset = dataset["test"]

    print(train_dataset)

    if model_args.apply_hyperadapter or model_args.apply_hyperlora or model_args.apply_hypertada:
        optimizer=torch.optim.AdamW([*model.parameters(), *model.hypernet.down_hypernet.parameters(), *model.hypernet.up_hypernet.parameters()],
            lr=training_args.learning_rate)
    scheduler=transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
        num_warmup_steps=0.06*len(train_dataset)*training_args.num_train_epochs,
        num_training_steps=len(train_dataset)*training_args.num_train_epochs)

    optimizers = optimizer, scheduler
    # 8. Initalize our trainer
    trainer = RobertaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        dataset_sizes=dataset_sizes,
        optimizers=optimizers
    )

    best_loss = 1000
    # 9. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        torch.save(trainer.model.hypernet.state_dict(), model_args.save_adapter_path)

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # 10. Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 11. Write Training Stats and push to hub.
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "embedding-alignment",
    }
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # elif data_args.push_adapter_to_hub:
    #     model.push_adapter_to_hub(
    #         data_args.adapter_repo_id,
    #         "tada_aave",
    #         organization=data_args.adapter_org_id,
    #         private=training_args.hub_private_repo,
    #         use_auth_token=model_args.use_auth_token,
    #         adapter_card_kwargs=kwargs,
    #         datasets_tag="glue",
    #     )
    # else:
    #     trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()